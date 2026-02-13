use crate::cell::Cell16;
use cell::{Cell32, Cell8};
use std::fmt::Debug;
use std::io::stdin;
use std::ops::{BitAnd, BitOr, BitXor, Index, Not, RangeBounds, Shl, Shr};

mod asm;
mod cell;

pub use asm::assemble;

fn mask_first<L, N>(
    length: L,
    n: N,
) -> <N as BitAnd<<<<N as Not>::Output as Shl<L>>::Output as Not>::Output>>::Output
where
    N: Default + Not + BitAnd<<<<N as Not>::Output as Shl<L>>::Output as Not>::Output>,
    <N as Not>::Output: Shl<L>,
    <<N as Not>::Output as Shl<L>>::Output: Not,
{
    n & !(!N::default() << length)
}

fn mask_range<L, N>(start: L, length: L, n: N) -> <<N as Shr<L>>::Output as BitAnd<<<<N as Not>::Output as Shl<L>>::Output as Not>::Output>>::Output
where
    N: Default + Not + Shr<L>,
    <N as Not>::Output: Shl<L>,
    <<N as Not>::Output as Shl<L>>::Output: Not,
    <N as Shr<L>>::Output: BitAnd<<<<N as Not>::Output as Shl<L>>::Output as Not>::Output>,
{
    (n >> start) & !(!N::default() << length)
}

fn has_one_at(n: u32, index: u32) -> bool {
    (n >> index) & 1 == 1
}

fn sign_extend_from(length: u32, n: u32) -> u32 {
    if has_one_at(n, length - 1) {
        n | (u32::MAX << length)
    } else {
        n
    }
}

pub struct Runtime<const MEMORY: usize> {
    program_counter: Cell32,
    i_registers: [Cell32; 32],
    memory: [Cell8; MEMORY],
}

impl<const MEMORY: usize> Default for Runtime<MEMORY> {
    fn default() -> Self {
        assert_eq!(MEMORY as u32 as usize, MEMORY);
        Self {
            program_counter: Cell32::default(),
            i_registers: [Cell32::default(); 32],
            memory: [Cell8::default(); MEMORY],
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ExecutionStatus {
    Done,
    EnvironmentCallExecuted,
    EnvironmentBreakExecuted,
}

#[derive(Debug, Eq, PartialEq)]
pub enum ExecutionError {
    BadOpcode(u32),
    BadInstruction(u32),
    Other,
}

impl<const MEMORY: usize> Runtime<MEMORY> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn write_to_memory<'a>(&mut self, start_address: usize, data: impl Into<&'a [Cell8]>) {
        let data = data.into();
        self.memory[start_address..start_address + data.len()].copy_from_slice(data);
    }

    pub fn read_memory<R>(&self, address_range: R) -> &[Cell8]
    where
        R: RangeBounds<usize>,
        [Cell8; MEMORY]: Index<R, Output = [Cell8]>,
    {
        &self.memory[address_range]
    }

    pub fn write_to_memory_8<'a>(&mut self, address: usize, data: impl Into<Cell8>) {
        self.memory[address] = data.into();
    }

    pub fn read_memory_8(&self, address: usize) -> Cell8 {
        self.memory[address]
    }

    pub fn write_to_memory_16<'a>(&mut self, start_address: usize, data: impl Into<Cell16>) {
        let data = <[Cell8; 2]>::from(data.into());
        self.memory[start_address..start_address + data.len()].copy_from_slice(&data);
    }

    pub fn read_memory_16(&self, start_address: usize) -> Cell16 {
        self.memory[start_address..start_address + 2]
            .try_into()
            .unwrap()
    }

    pub fn write_to_memory_32<'a>(&mut self, start_address: usize, data: impl Into<Cell32>) {
        let data = <[Cell8; 4]>::from(data.into());
        self.memory[start_address..start_address + data.len()].copy_from_slice(&data);
    }

    pub fn read_memory_32(&self, start_address: usize) -> Cell32 {
        self.memory[start_address..start_address + 4]
            .try_into()
            .unwrap()
    }

    pub fn write_to_i_register(&mut self, register_index: usize, data: impl Into<Cell32>) {
        if register_index > 0 {
            self.i_registers[register_index] = data.into();
        }
    }

    pub fn write_to_program_counter(&mut self, data: impl Into<Cell32>) {
        self.program_counter = data.into();
    }

    pub fn read_i_register(&self, register_index: usize) -> Cell32 {
        self.i_registers[register_index]
    }

    pub fn read_program_counter(&self) -> Cell32 {
        self.program_counter
    }

    pub fn step(&mut self) -> Result<ExecutionStatus, ExecutionError> {
        let program_counter = self.read_program_counter().unsigned();
        if program_counter % 2 != 0 {
            return Err(ExecutionError::Other);
        }
        if program_counter.overflowing_add(4).1 {
            return Err(ExecutionError::Other);
        }
        if program_counter.wrapping_add(4) > MEMORY as u32 {
            return Err(ExecutionError::Other);
        }
        let instruction = self.read_memory_32(program_counter as usize).unsigned();
        let opcode: u32 = mask_first(7, instruction);
        match opcode {
            0b0110011 => {
                // r-type arithmetic
                let funct3: u32 = mask_range(12, 3, instruction);
                let funct7: u32 = mask_range(25, 7, instruction);
                let rs1 = mask_range(15, 5, instruction) as usize;
                let rs2 = mask_range(20, 5, instruction) as usize;
                let rd = mask_range(7, 5, instruction) as usize;
                let rs1_value = self.read_i_register(rs1).unsigned();
                let rs2_value = self.read_i_register(rs2).unsigned();
                let rd_value = match (funct3, funct7) {
                    // add
                    (0x0, 0x00) => rs1_value.wrapping_add(rs2_value),
                    // sub
                    (0x0, 0x20) => rs1_value.wrapping_sub(rs2_value),
                    // xor
                    (0x4, 0x00) => rs1_value.bitxor(rs2_value),
                    // or
                    (0x6, 0x00) => rs1_value.bitor(rs2_value),
                    // and
                    (0x7, 0x00) => rs1_value.bitand(rs2_value),
                    // sll
                    (0x1, 0x00) => rs1_value.wrapping_shl(rs2_value),
                    // srl
                    (0x5, 0x00) => rs1_value.wrapping_shr(rs2_value),
                    // sra
                    (0x5, 0x20) => (rs1_value as i32).wrapping_shr(rs2_value) as u32,
                    // slt
                    (0x2, 0x00) => ((rs1_value as i32) < (rs2_value as i32)) as u32,
                    // sltu
                    (0x3, 0x00) => (rs1_value < rs2_value) as u32,
                    _ => return Err(ExecutionError::BadInstruction(instruction)),
                };
                self.write_to_program_counter(program_counter.wrapping_add(4));
                self.write_to_i_register(rd, rd_value);
                Ok(ExecutionStatus::Done)
            }
            0b0010011 => {
                // i-type arithmetic
                let funct3: u32 = mask_range(12, 3, instruction);
                let funct7: u32 = mask_range(25, 7, instruction);
                let rs1 = mask_range(15, 5, instruction) as usize;
                let rd = mask_range(7, 5, instruction) as usize;
                let immediate = sign_extend_from(12, mask_range(20, 12, instruction));
                let rs1_value = self.read_i_register(rs1).unsigned();
                let rd_value = match (funct3, funct7) {
                    // addi
                    (0x0, _) => rs1_value.wrapping_add(immediate),
                    // xori
                    (0x4, _) => rs1_value.bitxor(immediate),
                    // ori
                    (0x6, _) => rs1_value.bitor(immediate),
                    // andi
                    (0x7, _) => rs1_value.bitand(immediate),
                    // slli
                    (0x1, 0x00) => rs1_value.wrapping_shl(immediate),
                    // srli
                    (0x5, 0x00) => rs1_value.wrapping_shr(immediate),
                    // srai
                    (0x5, 0x20) => (rs1_value as i32).wrapping_shr(immediate) as u32,
                    // slti
                    (0x2, _) => ((rs1_value as i32) < (immediate as i32)) as u32,
                    // sltiu
                    (0x3, _) => (rs1_value < immediate) as u32,
                    _ => return Err(ExecutionError::BadInstruction(instruction)),
                };
                self.write_to_program_counter(program_counter.wrapping_add(4));
                self.write_to_i_register(rd, rd_value);
                Ok(ExecutionStatus::Done)
            }
            0b0000011 => {
                // load instructions (i-type)
                let funct3: u32 = mask_range(12, 3, instruction);
                let rs1 = mask_range(15, 5, instruction) as usize;
                let rd = mask_range(7, 5, instruction) as usize;
                let immediate = sign_extend_from(12, mask_range(20, 12, instruction));
                let rs1_value = self.read_i_register(rs1).unsigned();
                let address = rs1_value.wrapping_add(immediate) as usize;
                let rd_value = match funct3 {
                    // lb
                    0x0 => self.read_memory_8(address).signed() as i32 as u32,
                    // lh
                    0x1 => self.read_memory_16(address).signed() as i32 as u32,
                    // lw
                    0x2 => self.read_memory_32(address).unsigned(),
                    // lbu
                    0x4 => self.read_memory_8(address).unsigned() as u32,
                    // lhu
                    0x5 => self.read_memory_16(address).unsigned() as u32,
                    _ => return Err(ExecutionError::BadInstruction(instruction)),
                };
                self.write_to_program_counter(program_counter.wrapping_add(4));
                self.write_to_i_register(rd, rd_value);
                Ok(ExecutionStatus::Done)
            }
            0b0100011 => {
                // store instructions (s-type)
                let funct3: u32 = mask_range(12, 3, instruction);
                let rs1 = mask_range(15, 5, instruction) as usize;
                let rs2 = mask_range(20, 5, instruction) as usize;
                let immediate = sign_extend_from(7, mask_range(25, 7, instruction)) << 5
                    | mask_range(7, 5, instruction);
                let rs1_value = self.read_i_register(rs1).unsigned();
                let rs2_value = self.read_i_register(rs2).unsigned();
                let start_address = rs1_value.wrapping_add(immediate) as usize;
                match funct3 {
                    // sb
                    0x0 => self.write_to_memory_8(start_address, rs2_value as u8),
                    // sh
                    0x1 => self.write_to_memory_16(start_address, rs2_value as u16),
                    // sw
                    0x2 => self.write_to_memory_32(start_address, rs2_value),
                    _ => return Err(ExecutionError::BadInstruction(instruction)),
                }
                self.write_to_program_counter(program_counter.wrapping_add(4));
                Ok(ExecutionStatus::Done)
            }
            0b1100011 => {
                // conditional branches (b-type)
                let funct3: u32 = mask_range(12, 3, instruction);
                let rs1 = mask_range(15, 5, instruction) as usize;
                let rs2 = mask_range(20, 5, instruction) as usize;
                let immediate = sign_extend_from(1, mask_range(31, 1, instruction)) << 12
                    | mask_range(7, 1, instruction) << 11
                    | mask_range(25, 6, instruction) << 5
                    | mask_range(8, 4, instruction) << 1;
                let rs1_value = self.read_i_register(rs1).unsigned();
                let rs2_value = self.read_i_register(rs2).unsigned();
                let branch = match funct3 {
                    // beq
                    0x0 => rs1_value == rs2_value,
                    // bne
                    0x1 => rs1_value != rs2_value,
                    // blt
                    0x4 => (rs1_value as i32) < (rs2_value as i32),
                    // bge
                    0x5 => (rs1_value as i32) >= (rs2_value as i32),
                    // bltu
                    0x6 => rs1_value < rs2_value,
                    // bgeu
                    0x7 => rs1_value >= rs2_value,
                    _ => return Err(ExecutionError::BadInstruction(instruction)),
                };
                let program_counter = if branch {
                    program_counter.wrapping_add(immediate)
                } else {
                    program_counter.wrapping_add(4)
                };
                self.write_to_program_counter(program_counter);
                Ok(ExecutionStatus::Done)
            }
            0b1101111 => {
                // jal (j-type)
                let rd = mask_range(7, 5, instruction) as usize;
                let immediate = sign_extend_from(1, mask_range(31, 1, instruction)) << 20
                    | mask_range(12, 8, instruction) << 12
                    | mask_range(20, 1, instruction) << 11
                    | mask_range(25, 6, instruction) << 5
                    | mask_range(21, 4, instruction) << 1;
                self.write_to_i_register(rd, program_counter.wrapping_add(4));
                self.write_to_program_counter(program_counter.wrapping_add(immediate));
                Ok(ExecutionStatus::Done)
            }
            0b1100111 => {
                // jalr (i-type)
                let funct3: u32 = mask_range(12, 3, instruction);
                let rs1 = mask_range(15, 5, instruction) as usize;
                let rd = mask_range(7, 5, instruction) as usize;
                let immediate = sign_extend_from(12, mask_range(20, 12, instruction));
                let rs1_value = self.read_i_register(rs1).unsigned();
                if funct3 != 0x0 {
                    return Err(ExecutionError::BadInstruction(instruction));
                }
                self.write_to_i_register(rd, program_counter.wrapping_add(4));
                self.write_to_program_counter(rs1_value.wrapping_add(immediate));
                Ok(ExecutionStatus::Done)
            }
            0b0110111 => {
                // lui (u-type)
                let rd = mask_range(7, 5, instruction) as usize;
                let immediate = mask_range(12, 20, instruction) << 12;
                self.write_to_i_register(rd, immediate);
                self.write_to_program_counter(program_counter.wrapping_add(4));
                Ok(ExecutionStatus::Done)
            }
            0b0010111 => {
                // auipc (u-type)
                let rd = mask_range(7, 5, instruction) as usize;
                let immediate = mask_range(12, 20, instruction) << 12;
                self.write_to_i_register(rd, program_counter.wrapping_add(immediate));
                self.write_to_program_counter(program_counter.wrapping_add(4));
                Ok(ExecutionStatus::Done)
            }
            0b1110011 => {
                // environment instructions (i-type)
                let funct3: u32 = mask_range(12, 3, instruction);
                let immediate: u32 = mask_range(20, 12, instruction);
                if funct3 != 0x0 {
                    return Err(ExecutionError::BadInstruction(instruction));
                }
                match immediate {
                    // ecall
                    0x0 => Ok(ExecutionStatus::EnvironmentCallExecuted),
                    // ebreak
                    0x1 => Ok(ExecutionStatus::EnvironmentBreakExecuted),
                    _ => Err(ExecutionError::BadInstruction(instruction)),
                }
            }
            _ => Err(ExecutionError::BadOpcode(opcode)),
        }
    }

    pub fn run(&mut self) -> Result<ExecutionStatus, ExecutionError> {
        loop {
            let status = self.step()?;
            match status {
                ExecutionStatus::Done => continue,
                ExecutionStatus::EnvironmentCallExecuted => break Ok(status),
                ExecutionStatus::EnvironmentBreakExecuted => break Ok(status),
            }
        }
    }
}

impl<const MEMORY: usize> Runtime<MEMORY> {
    fn dump_registers(&self) {
        println!("reg  hex.....  dec");
        let program_counter = self.read_program_counter();
        println!(
            " pc  {:08x}  {}",
            program_counter.unsigned(),
            program_counter.signed()
        );
        for index in 0..32 {
            let value = self.read_i_register(index);
            if index < 10 {
                print!(" ");
            }
            print!("x{index}  ");
            print!("{:08x}  ", value.unsigned());
            println!("{}", value.signed());
        }
        println!();
    }

    fn dump_memory(&self) {
        const ROW_SIZE: usize = 0x10;
        print!("   ");
        for column_index in 0..ROW_SIZE {
            print!("{column_index:02x} ");
        }
        println!();
        let rows = MEMORY.div_ceil(ROW_SIZE);
        for row_index in 0..rows {
            print!("{row_index:02x} ");
            for column_index in 0..ROW_SIZE {
                let address = row_index * ROW_SIZE + column_index;
                if address >= MEMORY {
                    break;
                }
                let data = self.read_memory_8(address).unsigned();
                print!("{data:02x} ");
            }
            println!();
        }
        println!();
    }
}

fn main() {
    let code = "\
jal x0, main

# CONTRACT:
# before:
# - x10 = pointer to first byte of string
# after:
# - x10 = pointer to terminal null byte of string
# - x5 undefined
string_end:
# x10 = pointer to first byte
# x10 = pointer to current byte
string_end__continue:
lb x5, 0(x10)
beq x5, x0, string_end__break
addi x10, x10, 1
jal x0, string_end__continue
string_end__break:
# x10 = pointer to last byte
jalr x0, x1, 0

# CONTRACT:
# before:
# - x10 = in-bounds pointer
# after:
# - x10 unchanged
# - x11 undefined
# side effects:
# - writes string acquired from user to x10
request_input:
# x10 = pointer to future first byte of string
addi x11, x10, 0
addi x10, x0, 3
# does the user-input and string-storing things
ecall
addi x10, x11, 0
jalr x0, x1, 0

# CONTRACT:
# before:
# - x10 = pointer to string
# after:
# - x10 unchanged
# - x11 undefined
# side effects:
# - prints string at x10
print_string:
# x10 = pointer to first byte of string
addi x11, x10, 0
addi x10, x0, 2
# does the printing
ecall
addi x10, x11, 0
jalr x0, x1, 0

# CONTRACT:
# before:
# - x10 = pointer to string
# - x11 = valid string byte (i.e. lower 8 bits not all 0)
# after:
# - x10 unchanged
# - x11 unchanged
# - x5 undefined
# - x6 undefined
# - x7 undefined
# side effects:
# - appends lower byte of x11 to string at x10 (preserving null termination)
append_to_string:
# save return address and x10
addi x6, x1, 0
addi x7, x10, 0
# x10 = pointer to first byte of string
# x11 = byte to append
# string_end doesn't touch x11 (but it does touch x5!)
jal x1, string_end
# x10 = pointer to last byte of string
# overwrite terminal null
sb x11, 0(x10)
# restore null termination
sb x0, 1(x10)
# restore return address and x10
addi x1, x6, 0
addi x10, x7, 0
# return
jalr x0, x1, 0

# CONTRACT:
# before:
# - x10 = pointer to string
# - x11 = valid string byte (i.e. lower 8 bits not all 0)
# after:
# - x10 unchanged
# - x11 unchanged
# - x5 undefined
# - x6 undefined
# - x7 undefined
# - x28 undefined
# side effects:
# - prepends lower byte of x11 to string at x10 (preserving null termination)
prepend_to_string:
# save return address and x10
addi x6, x1, 0
addi x7, x10, 0
# x10 = pointer to first byte of string
# x11 = byte to prepend
# string_end doesn't touch x11
jal x1, string_end
# move all bytes one up
# x10 = pointer to terminal null byte of string
# x10 = pointer to current byte
prepend_to_string__continue:
# move byte one up
lb x28, 0(x10)
sb x28, 1(x10)
# decrement x10
addi x10, x10, -1
bge x10, x7, prepend_to_string__continue
# x10 = pointer to first byte of string - 1
# (because only then does the loop break)
addi x10, x10, 1
# prepend byte
sb x11, 0(x10)
# restore return address
addi x1, x6, 0
# return
jalr x0, x1, 0

ask_for_name:
# save return address
addi x29, x1, 0
# x10 = string address
addi x10, x0, 1024
# initialize as empty string
sw x0, 0(x10)
# append chars one by one
# N
addi x11, x0, 0x4e
jal x1, append_to_string
# a
addi x11, x0, 0x61
jal x1, append_to_string
# m
addi x11, x0, 0x6d
jal x1, append_to_string
# e
addi x11, x0, 0x65
jal x1, append_to_string
# ?
addi x11, x0, 0x3f
jal x1, append_to_string
# newline
addi x11, x0, 0x0a
jal x1, append_to_string
# print the string
jal x1, print_string

jal x1, request_input
# space
addi x11, x0, 0x20
jal x1, prepend_to_string
# ,
addi x11, x0, 0x2c
jal x1, prepend_to_string
# o
addi x11, x0, 0x6f
jal x1, prepend_to_string
# l
addi x11, x0, 0x6c
jal x1, prepend_to_string
# l
addi x11, x0, 0x6c
jal x1, prepend_to_string
# e
addi x11, x0, 0x65
jal x1, prepend_to_string
# H
addi x11, x0, 0x48
jal x1, prepend_to_string
# newline
addi x11, x0, 0x0a
jal x1, prepend_to_string
# [name]
# !
addi x11, x0, 0x21
jal x1, append_to_string
# newline
addi x11, x0, 0x0a
jal x1, append_to_string
# newline
addi x11, x0, 0x0a
jal x1, append_to_string
# print the string
jal x1, print_string

# restore return address
addi x1, x29, 0
# return
jalr x0, x1, 0

main:
jal x1, ask_for_name
ebreak
";
    let instructions = assemble(code).unwrap();

    println!("instructions hex dump:");
    for (index, &instruction) in instructions.iter().enumerate() {
        println!("{:08x}  {instruction:08x}", index * 4);
    }
    println!();
    println!();

    let mut runtime = Runtime::<2048>::new();
    for (index, instruction) in instructions.into_iter().enumerate() {
        runtime.write_to_memory_32(index * 4, instruction);
    }

    loop {
        match runtime.run() {
            Ok(status) => {
                match status {
                    ExecutionStatus::Done => {
                        println!("program finished");
                        break;
                    }
                    ExecutionStatus::EnvironmentCallExecuted => {
                        let a0_value = runtime.read_i_register(10);
                        match a0_value.unsigned() {
                            0 => {
                                // print registers
                                runtime.dump_registers();
                            }
                            1 => {
                                // print register specified by a1
                                let a1_value = runtime.read_i_register(11);
                                let register_index = a1_value.unsigned() as usize;
                                if register_index > 32 {
                                    println!("bad register index {register_index} requested");
                                } else {
                                    let register_value = runtime.read_i_register(register_index);
                                    print!("x{register_index} = ");
                                    print!("{:08x}, ", register_value.unsigned());
                                    println!("{}", register_value.signed());
                                }
                            }
                            2 => {
                                // print null-terminated string at memory address in a1
                                let a1_value = runtime.read_i_register(11);
                                let address = a1_value.unsigned() as usize;
                                if address >= runtime.memory.len() {
                                    println!("bad memory address {address}");
                                } else {
                                    let mut string_bytes = Vec::new();
                                    let mut running_address = address;
                                    loop {
                                        let byte = runtime.read_memory_8(running_address);
                                        if byte.unsigned() == 0 {
                                            break;
                                        }
                                        string_bytes.push(byte.unsigned());
                                        running_address = running_address.wrapping_add(1);
                                        running_address %= runtime.memory.len();
                                    }
                                    match String::from_utf8(string_bytes) {
                                        Ok(string) => {
                                            print!("{string}");
                                        }
                                        Err(_error) => {
                                            println!("string not valid UTF8");
                                        }
                                    }
                                }
                            }
                            3 => {
                                // store user-input null-terminated string at memory address in a1
                                // user input is trimmed before being written to memory
                                let a1_value = runtime.read_i_register(11);
                                let address = a1_value.unsigned() as usize;
                                if address >= runtime.memory.len() {
                                    println!("bad memory address {address}");
                                } else {
                                    let mut buffer = String::new();
                                    if let Err(error) = stdin().read_line(&mut buffer) {
                                        println!("couldn't read user input: {error:?}");
                                        // store an empty string instead
                                    }
                                    let string = buffer.trim();
                                    let bytes = string.as_bytes();
                                    let mut running_address = address;
                                    for &byte in bytes {
                                        if byte == 0 {
                                            println!("null byte in user input");
                                            // ignore the rest of the string
                                            break;
                                        }
                                        runtime.write_to_memory_8(running_address, byte);
                                        running_address = running_address.wrapping_add(1);
                                        running_address %= runtime.memory.len();
                                    }
                                    runtime.write_to_memory_8(running_address, 0u8);
                                }
                            }
                            other => {
                                println!("unrecognized ecall request code: {other}");
                            }
                        }
                        let program_counter = runtime.read_program_counter();
                        runtime
                            .write_to_program_counter(program_counter.unsigned().wrapping_add(4));
                    }
                    ExecutionStatus::EnvironmentBreakExecuted => {
                        println!("environment break executed, exiting program");
                        break;
                        // println!("environment break executed");
                        // let program_counter = runtime.read_program_counter();
                        // runtime
                        //     .write_to_program_counter(program_counter.unsigned().wrapping_add(4));
                    }
                }
            }
            Err(error) => {
                println!("execution error encountered: {error:?}, exiting program");
                println!();
                runtime.dump_registers();
                runtime.dump_memory();
                break;
            }
        }
    }
}
