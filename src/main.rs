use anyhow::{Context, Result};
use clap::Parser;
use kvm_bindings::kvm_segment;
use kvm_bindings::kvm_userspace_memory_region;
use kvm_ioctls::Kvm;
use kvm_ioctls::VcpuExit;
use linux_loader::cmdline::Cmdline;
use linux_loader::configurator::linux::LinuxBootConfigurator;
use linux_loader::configurator::{BootConfigurator, BootParams};
use linux_loader::loader::bootparam::boot_params;
use linux_loader::loader::elf::Elf;
use linux_loader::loader::{KernelLoader, load_cmdline};
use std::fs::File;
use std::io::{self, Read};
use std::os::unix::io::AsRawFd;
use std::sync::{Arc, Mutex};
use std::thread;
use vm_memory::Bytes;
use vm_memory::{Address, GuestAddress, GuestMemoryBackend, GuestMemoryMmap, GuestMemoryRegion};
use vm_superio::Trigger;
use vm_superio::serial::Serial;
use vmm_sys_util::eventfd::EventFd;

const HIMEM_START: u64 = 0x10_0000; // 1 MiB — where the 64-bit kernel goes
const CMDLINE_START: u64 = 0x2_0000;

const ZERO_PAGE_START: u64 = 0x7000;

// e820 entry types and the boot protocol magic numbers.
const KERNEL_BOOT_FLAG_MAGIC: u16 = 0xaa55;
const KERNEL_HDR_MAGIC: u32 = 0x5372_6448; // "HdrS"
const KERNEL_LOADER_OTHER: u8 = 0xff;
const E820_RAM: u32 = 1;
const EBDA_START: u64 = 0x9_fc00; // top of usable low memory
const HIMEM_START_ADDR: u64 = 0x10_0000; // 1 MiB

const BOOT_GDT_OFFSET: u64 = 0x500;
const PML4_START: u64 = 0x9000;
const PDPTE_START: u64 = 0xa000;
const PDE_START: u64 = 0xb000;

const X86_CR0_PE: u64 = 0x1;
const X86_CR0_PG: u64 = 0x8000_0000;
const X86_CR4_PAE: u64 = 0x20;
const EFER_LME: u64 = 0x100;
const EFER_LMA: u64 = 0x400;

const COM1_BASE: u16 = 0x3f8;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    kernel: String,
    #[arg(long, default_value_t = 512)]
    mem_mib: usize,
    #[arg(long, default_value = "console=ttyS0 reboot=k panic=1 rdinit=/init")]
    cmdline: String,
    #[arg(long)]
    initramfs: String,
}

fn add_e820_entry(params: &mut boot_params, addr: u64, size: u64, mem_type: u32) {
    let idx = params.e820_entries as usize;
    params.e820_table[idx].addr = addr;
    params.e820_table[idx].size = size;
    params.e820_table[idx].r#type = mem_type;
    params.e820_entries += 1;
}

fn gdt_entry(flags: u16, base: u32, limit: u32) -> u64 {
    ((u64::from(base) & 0xff00_0000) << 32)
        | ((u64::from(flags) & 0x0000_f0ff) << 40)
        | ((u64::from(limit) & 0x000f_0000) << 32)
        | ((u64::from(base) & 0x00ff_ffff) << 16)
        | (u64::from(limit) & 0x0000_ffff)
}

// Helper: turn a GDT entry into the kvm_segment KVM wants.
fn seg_from_gdt(entry: u64, table_index: u8) -> kvm_segment {
    let base = (((entry >> 16) & 0xffffff) | ((entry >> 32) & 0xff00_0000)) as u64;
    let limit = (((entry) & 0xffff) | ((entry >> 32) & 0xf_0000)) as u32;
    let flags = ((entry >> 40) & 0xf0ff) as u16;
    kvm_segment {
        base,
        limit,
        selector: (table_index as u16) * 8,
        type_: (flags & 0xf) as u8,
        present: ((flags >> 7) & 1) as u8,
        dpl: ((flags >> 5) & 3) as u8,
        db: ((flags >> 14) & 1) as u8,
        s: ((flags >> 4) & 1) as u8,
        l: ((flags >> 13) & 1) as u8,
        g: ((flags >> 15) & 1) as u8,
        avl: ((flags >> 12) & 1) as u8,
        ..Default::default()
    }
}

fn setup_long_mode(
    vcpu: &kvm_ioctls::VcpuFd,
    guest_mem: &GuestMemoryMmap,
    entry: u64,
) -> Result<()> {
    // --- minimal identity-mapped page tables ---
    // PML4[0] -> PDPTE,  PDPTE[0] -> PDE,  PDE entries -> 2 MiB pages.
    guest_mem.write_obj(PDPTE_START | 0x03, GuestAddress(PML4_START))?;
    guest_mem.write_obj(PDE_START | 0x03, GuestAddress(PDPTE_START))?;
    for i in 0..512u64 {
        // 0x83 = present | writable | huge-page(2 MiB)
        guest_mem.write_obj((i << 21) | 0x83, GuestAddress(PDE_START + i * 8))?;
    }

    let mut sregs = vcpu.get_sregs()?;

    // --- GDT: null, code, data ---
    let gdt = [
        gdt_entry(0, 0, 0),            // null
        gdt_entry(0xa09b, 0, 0xfffff), // code: present, exec, 64-bit (L bit)
        gdt_entry(0xc093, 0, 0xfffff), // data: present, writable
    ];
    for (i, entry) in gdt.iter().enumerate() {
        guest_mem.write_obj(*entry, GuestAddress(BOOT_GDT_OFFSET + (i as u64) * 8))?;
    }
    sregs.gdt.base = BOOT_GDT_OFFSET;
    sregs.gdt.limit = (gdt.len() * 8 - 1) as u16;

    let code_seg = seg_from_gdt(gdt[1], 1);
    let data_seg = seg_from_gdt(gdt[2], 2);
    sregs.cs = code_seg;
    sregs.ds = data_seg;
    sregs.es = data_seg;
    sregs.fs = data_seg;
    sregs.gs = data_seg;
    sregs.ss = data_seg;

    // --- the actual mode switch ---
    sregs.cr3 = PML4_START;
    sregs.cr4 |= X86_CR4_PAE;
    sregs.cr0 |= X86_CR0_PE | X86_CR0_PG;
    sregs.efer |= EFER_LME | EFER_LMA;

    vcpu.set_sregs(&sregs)?;

    // --- general registers: entry point + boot_params pointer ---
    let mut regs = vcpu.get_regs()?;
    regs.rflags = 0x2;
    regs.rip = entry;
    regs.rsi = 0x7000; // ZERO_PAGE_START — kernel reads boot_params from rsi
    regs.rsp = 0x8ff0;
    vcpu.set_regs(&regs)?;
    Ok(())
}

// vm-superio needs a Trigger; EventFd isn't one, so wrap it.
struct EventFdTrigger(EventFd);
impl Trigger for EventFdTrigger {
    type E = io::Error;
    fn trigger(&self) -> Result<(), Self::E> {
        self.0.write(1)
    }
}

fn set_raw_mode() -> libc::termios {
    let fd = io::stdin().as_raw_fd();
    unsafe {
        let mut termios: libc::termios = std::mem::zeroed();
        libc::tcgetattr(fd, &mut termios);
        let original = termios; // copy the pristine settings before we mangle them
        libc::cfmakeraw(&mut termios);
        libc::tcsetattr(fd, libc::TCSANOW, &termios);
        original // return them to the caller
    }
}

fn restore_mode(original: &libc::termios) {
    let fd = io::stdin().as_raw_fd();
    unsafe {
        libc::tcsetattr(fd, libc::TCSANOW, original);
    }
}

struct RawMode(libc::termios);

impl RawMode {
    fn new() -> Self {
        RawMode(set_raw_mode())
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        restore_mode(&self.0)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mem_size = args.mem_mib << 20;

    let _raw = RawMode::new();
    let original = _raw.0;

    ctrlc::set_handler(move || {
        restore_mode(&original);
        eprintln!("\nterminated, terminal restored");
        std::process::exit(130);
    })
    .expect("installing signal handler");

    let kvm = Kvm::new().context("opening /dev/kvm")?;
    let vm = kvm.create_vm().context("creating VM")?;
    vm.create_irq_chip().context("creating virt irq chip")?;

    // 1. Guest RAM as a vm-memory object.
    let guest_mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0x0), mem_size)])
        .context("mmaping guest memory")?;

    // 2. Hand each region to KVM.
    for (i, region) in guest_mem.iter().enumerate() {
        let mr = kvm_userspace_memory_region {
            slot: i as u32,
            guest_phys_addr: region.start_addr().raw_value(),
            memory_size: region.len() as u64,
            userspace_addr: guest_mem
                .get_host_address(region.start_addr())
                .context("getting region host addr")? as u64,
            flags: 0,
        };
        unsafe {
            vm.set_user_memory_region(mr)
                .context("setting guest user memory region")?
        }
    }

    // 3. Load ELF kernel
    let mut kernel = File::open(args.kernel).context("opening kernel file")?;
    let load = Elf::load(
        &guest_mem,
        None,
        &mut kernel,
        Some(GuestAddress(HIMEM_START)),
    )
    .context("loading elf kernel onto guest")?;
    println!("kernel entry: {:#x}", load.kernel_load.raw_value());

    // Load initramfs and place it in guest memory
    let mut initrd = Vec::new();
    File::open(args.initramfs)
        .context("opening initramfs")?
        .read_to_end(&mut initrd)
        .context("reading initramfs")?;
    let initrd_size = initrd.len();

    let initrd_addr = (mem_size - initrd_size) & !0xfff_usize; // round down to 4 KiB
    guest_mem
        .write_slice(&initrd, GuestAddress(initrd_addr as u64))
        .context("writing initrd address in guest")?;

    println!("initramfs: {} bytes at {:#x}", initrd_size, initrd_addr);

    // 4. Command line into guest memory.
    let mut cmdline = Cmdline::new(0x10000).context("allocating cmdline buf")?;
    cmdline
        .insert_str(args.cmdline)
        .context("writing cmdline to buf")?;
    load_cmdline(&guest_mem, GuestAddress(CMDLINE_START), &cmdline)
        .context("loading cmdline buf to guest memory")?;

    // Build boot params
    let mut params = boot_params::default();
    params.hdr.type_of_loader = KERNEL_LOADER_OTHER;
    params.hdr.boot_flag = KERNEL_BOOT_FLAG_MAGIC;
    params.hdr.header = KERNEL_HDR_MAGIC;
    params.hdr.cmd_line_ptr = CMDLINE_START as u32;
    params.hdr.cmdline_size = cmdline
        .as_cstring()
        .context("converting cmdline buf to cstring")?
        .as_bytes()
        .len() as u32
        + 1;
    params.hdr.ramdisk_image = initrd_addr as u32;
    params.hdr.ramdisk_size = initrd_size as u32;

    // Memory map: low RAM (below the BIOS/EBDA area), then RAM from 1 MiB up.
    add_e820_entry(&mut params, 0, EBDA_START, E820_RAM);
    add_e820_entry(
        &mut params,
        HIMEM_START_ADDR,
        (mem_size as u64) - HIMEM_START_ADDR,
        E820_RAM,
    );

    // Then write the constructed bootparams to guest memory
    LinuxBootConfigurator::write_bootparams::<GuestMemoryMmap>(
        &BootParams::new(&params, GuestAddress(ZERO_PAGE_START)),
        &guest_mem,
    )
    .context("mmaping boot params to guest")?;

    let mut vcpu = vm.create_vcpu(0).context("creating vcpu in guest")?;

    // CPUID — let the kernel's feature checks pass.
    let kvm_cpuid = kvm
        .get_supported_cpuid(kvm_bindings::KVM_MAX_CPUID_ENTRIES)
        .context("getting supported cpuid")?;
    vcpu.set_cpuid2(&kvm_cpuid)
        .context("setting cpuid2 on guest")?;

    setup_long_mode(&vcpu, &guest_mem, load.kernel_load.raw_value())
        .context("setting up guest long mode")?;

    // Create serial device and interrupt FD, then register with KVM on COM1 line (IRQ 4)
    let com1_evt =
        EventFd::new(libc::EFD_NONBLOCK).context("creating event fd for serial device")?;
    vm.register_irqfd(&com1_evt, 4)
        .context("registering event fd as irq in guest")?;

    let serial = Arc::new(Mutex::new(Serial::new(
        EventFdTrigger(
            com1_evt
                .try_clone()
                .context("cloning event fd for serial device")?,
        ),
        io::stdout(),
    )));

    // Input handling
    let serial_input = Arc::clone(&serial);
    thread::spawn(move || {
        let mut buf = [0u8; 1];
        loop {
            if io::stdin().read(&mut buf).unwrap() == 1 {
                serial_input
                    .lock()
                    .unwrap()
                    .enqueue_raw_bytes(&buf)
                    .unwrap();
            }
        }
    });

    // Finally, the kvm run loop
    loop {
        match vcpu.run().expect("vcpu run failed") {
            VcpuExit::IoOut(addr, data) => {
                if (COM1_BASE..COM1_BASE + 8).contains(&addr) {
                    serial
                        .lock()
                        .unwrap()
                        .write((addr - COM1_BASE) as u8, data[0])
                        .context("writing to vcpu")?;
                }
            }
            VcpuExit::IoIn(addr, data) => {
                if (COM1_BASE..COM1_BASE + 8).contains(&addr) {
                    data[0] = serial.lock().unwrap().read((addr - COM1_BASE) as u8);
                }
            }
            VcpuExit::Hlt => {
                println!("\nguest halted");
                break;
            }
            VcpuExit::Shutdown => {
                println!("\nguest shutdown");
                break;
            }
            other => {
                println!("unhandled exit: {:?}", other);
            }
        }
    }
    Ok(())
}
