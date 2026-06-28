use kvm_bindings::kvm_userspace_memory_region;
use kvm_ioctls::{Kvm, VcpuExit};

fn main() {
    let mem_size = 0xf000;
    let guest_addr: u64 = 0x1000;

    // Real-mode machine code: compute 2+3, turn it into ASCII, write it to
    // port 0x3f8 (COM1), write a newline, then halt.
    let code: &[u8] = &[
        0xba, 0xf8, 0x03, // mov  dx, 0x3f8
        0x00, 0xd8, // add  al, bl
        0x04, b'0', // add  al, '0'
        0xee, // out  dx, al
        0xb0, b'\n', // mov  al, '\n'
        0xee,  // out  dx, al
        0xf4,  // hlt
    ];

    let kvm = Kvm::new().unwrap();
    let vm = kvm.create_vm().unwrap();

    let host_mem = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            mem_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANONYMOUS | libc::MAP_SHARED | libc::MAP_NORESERVE,
            -1,
            0,
        ) as *mut u8
    };

    // Tell KVM: "this host memory IS guest physical memory starting at guest_addr."
    let region = kvm_userspace_memory_region {
        slot: 0,
        guest_phys_addr: guest_addr,
        memory_size: mem_size as u64,
        userspace_addr: host_mem as u64,
        flags: 0,
    };
    unsafe {
        vm.set_user_memory_region(region).unwrap();
    }

    // Copy our code into the start of that region (= guest phys 0x1000).
    unsafe {
        std::slice::from_raw_parts_mut(host_mem, mem_size)[..code.len()].copy_from_slice(code);
    }
    let mut vcpu = vm.create_vcpu(0).unwrap();

    // Real mode, code segment based at 0, instruction pointer at our code.
    let mut sregs = vcpu.get_sregs().unwrap();
    sregs.cs.base = 0;
    sregs.cs.selector = 0;
    vcpu.set_sregs(&sregs).unwrap();

    let mut regs = vcpu.get_regs().unwrap();
    regs.rip = guest_addr; // start executing at 0x1000
    regs.rax = 2;
    regs.rbx = 3;
    regs.rflags = 2; // bit 1 is reserved-and-always-set; KVM rejects 0
    vcpu.set_regs(&regs).unwrap();

    // The run / exit loop — the heart of every VMM.
    loop {
        match vcpu.run().expect("vcpu run failed") {
            VcpuExit::IoOut(port, data) => {
                print!("[guest wrote to port {:#x}] {}", port, data[0] as char);
            }
            VcpuExit::Hlt => {
                println!("\nguest halted - done.");
                break;
            }
            other => panic!("unexpected VM exit: {:?}", other),
        }
    }
}
