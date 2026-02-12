use crate::cell::Cell16;
use cell::{Cell32, Cell8};
use std::fmt::Debug;
use std::ops::{BitAnd, BitOr, BitXor, Index, Not, RangeBounds, Shl, Shr};

mod cell;

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
                    // sll
                    (0x1, 0x00) => rs1_value.wrapping_shl(immediate),
                    // srl
                    (0x5, 0x00) => rs1_value.wrapping_shr(immediate),
                    // sra
                    (0x5, 0x20) => (rs1_value as i32).wrapping_shr(immediate) as u32,
                    // slt
                    (0x2, _) => ((rs1_value as i32) < (immediate as i32)) as u32,
                    // sltu
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
                self.write_to_program_counter(
                    program_counter
                        .wrapping_add(rs1_value)
                        .wrapping_add(immediate),
                );
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

fn main() {
    /*
    convert_null_terminated_to_pascal:
      # a0 = pointer to first byte
      # t0 = pointer to current byte
      mv t0, a0
    start_of_loop:
      # t1 = current byte
      lb t1, 0(t0)
      # break loop if null
      beq t1, zero, end_of_loop
      # increment current byte pointer
      addi t0, t0, 1
      j start_of_loop
    end_of_loop:
      # t0 = pointer to last byte (= null)
      # t2 = length of string
      sub t2, t0, a0
    start_of_loop_2:
      # break loop if start of string has been reached
      beq t0, a0, end_of_loop_2
      # t1 = current byte
      lb t1, -1(t0)
      # move it one forward
      sb t1, 0(t0)
      # decrement current byte pointer
      addi t0, t0, -1
      j start_of_loop_2
    end_of_loop_2:
      sb t2, 0(a0)
      jr ra
     */

    let mut runtime = Runtime::<0x200>::new();
    let hex_dump: [u32; _] = [
        0x00050293,
        0x00028303,
        0x00030663,
        0x00128293,
        0xff5ff06f,
        0x40a283b3,
        0x00a28a63,
        0xfff28303,
        0x00628023,
        0xfff28293,
        0xff1ff06f,
        0x00750023,
        // ecall instead of jr ra
        0b0000000_00000_00000_000_00000_1110011,
    ];
    for (index, instruction) in hex_dump.into_iter().enumerate() {
        runtime.write_to_memory_32(index * 4, instruction);
    }
    let data: [u8; _] = [
        11,
        22,
        33,
        44,
        0,
    ];
    for (index, byte) in data.into_iter().enumerate() {
        runtime.write_to_memory_8(0x100 + index, byte);
    }
    runtime.write_to_i_register(10, 0x100);
    assert_eq!(runtime.run(), Ok(ExecutionStatus::EnvironmentCallExecuted));
    assert_eq!(runtime.read_program_counter(), (12 * 4).into());
    let result = runtime.read_memory(0x100..(0x100 + data.len()));
    println!("{result:?}");
}
