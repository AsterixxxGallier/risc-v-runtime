use crate::{mask_first, mask_range};
use pest::iterators::{Pair, Pairs};
use pest::Parser;
use pest_derive::Parser;
use std::collections::HashMap;

#[derive(Parser)]
#[grammar = "syntax.pest"]
struct AsmParser;

fn register(pair: Pair<Rule>) -> u32 {
    let number_part = &pair.as_str()[1..];
    number_part.parse().unwrap()
}

fn immediate(pair: Pair<Rule>) -> Result<u32, String> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::dec_imm => {
            let signed: i32 = inner.as_str().parse().unwrap();
            if signed < -(1 << 11) || signed >= (1 << 11) {
                Err(format!("immediate {signed} out of range"))
            } else {
                Ok(signed as u32)
            }
        }
        _ => unreachable!(),
    }
}

fn unsigned_immediate(pair: Pair<Rule>) -> Result<u32, String> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::dec_uimm => {
            let unsigned: u32 = inner.as_str().parse().unwrap();
            if unsigned >= (1 << 5) {
                Err(format!("immediate {unsigned} out of range"))
            } else {
                Ok(unsigned)
            }
        }
        _ => unreachable!(),
    }
}

fn upper_immediate(pair: Pair<Rule>) -> Result<u32, String> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::dec_upimm => {
            let unsigned: u32 = inner.as_str().parse().unwrap();
            if unsigned >= (1 << 20) {
                Err(format!("immediate {unsigned} out of range"))
            } else {
                Ok(unsigned)
            }
        }
        _ => unreachable!(),
    }
}

fn immediate_register(pair: Pair<Rule>) -> Result<(u32, u32), String> {
    let mut pairs = pair.into_inner();
    let immediate = immediate(pairs.next().unwrap())?;
    let register = register(pairs.next().unwrap());
    Ok((immediate, register))
}

fn load_instruction(mut pairs: Pairs<Rule>, funct3: u32) -> Result<u32, String> {
    let opcode: u32 = 0b0000011;
    let rd = register(pairs.next().unwrap());
    let (imm, rs) = immediate_register(pairs.next().unwrap())?;
    let imm = mask_first(12, imm);
    Ok(opcode | (rd << 7) | (funct3 << 12) | (rs << 15) | (imm << 20))
}

fn immediate_arithmetic_instruction(
    mut pairs: Pairs<Rule>,
    funct3: u32,
    funct7: Option<u32>,
) -> Result<u32, String> {
    let opcode: u32 = 0b0010011;
    let rd = register(pairs.next().unwrap());
    let rs = register(pairs.next().unwrap());
    let imm = if funct7.is_some() {
        mask_first(5, unsigned_immediate(pairs.next().unwrap())?)
    } else {
        mask_first(12, immediate(pairs.next().unwrap())?)
    };
    let funct7 = funct7.unwrap_or(0);
    Ok(opcode | (rd << 7) | (funct3 << 12) | (rs << 15) | (imm << 20) | (funct7 << 25))
}

fn lui_or_auipc(mut pairs: Pairs<Rule>, opcode: u32) -> Result<u32, String> {
    let rd = register(pairs.next().unwrap());
    let imm = upper_immediate(pairs.next().unwrap())?;
    Ok(opcode | (rd << 7) | (imm << 12))
}

fn store_instruction(mut pairs: Pairs<Rule>, funct3: u32) -> Result<u32, String> {
    let opcode: u32 = 0b0100011;
    let rs2 = register(pairs.next().unwrap());
    let (imm, rs1) = immediate_register(pairs.next().unwrap())?;
    let imm_lower = mask_first(5, imm);
    let imm_upper = mask_range(5, 7, imm);
    Ok(opcode | (imm_lower << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm_upper << 25))
}

fn register_arithmetic_instruction(
    mut pairs: Pairs<Rule>,
    funct3: u32,
    funct7: u32,
) -> Result<u32, String> {
    let opcode: u32 = 0b0110011;
    let rd = register(pairs.next().unwrap());
    let rs1 = register(pairs.next().unwrap());
    let rs2 = register(pairs.next().unwrap());
    Ok(opcode | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (funct7 << 25))
}

fn branch_instruction(
    labels: &HashMap<String, u32>,
    position: u32,
    mut pairs: Pairs<Rule>,
    funct3: u32,
) -> Result<u32, String> {
    let opcode: u32 = 0b1100011;
    let rs1 = register(pairs.next().unwrap());
    let rs2 = register(pairs.next().unwrap());
    let label = pairs.next().unwrap().as_str();
    let label_position = labels
        .get(label)
        .copied()
        .ok_or(format!("couldn't find label {label}"))?;
    let offset = ((label_position as i32 - position as i32) * 4) as u32;
    let imm_lower = mask_first(5, offset) | mask_range(11, 1, offset);
    let imm_upper = mask_range(5, 6, offset) | (mask_range(12, 1, offset) << 6);
    Ok(opcode | (imm_lower << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20) | (imm_upper << 25))
}

fn jalr(mut pairs: Pairs<Rule>) -> Result<u32, String> {
    let opcode: u32 = 0b1100111;
    let funct3: u32 = 0x0;
    let rd = register(pairs.next().unwrap());
    let rs = register(pairs.next().unwrap());
    let imm = mask_first(12, immediate(pairs.next().unwrap())?);
    Ok(opcode | (rd << 7) | (funct3 << 12) | (rs << 15) | (imm << 20))
}

fn jal(
    labels: &HashMap<String, u32>,
    position: u32,
    mut pairs: Pairs<Rule>,
) -> Result<u32, String> {
    let opcode: u32 = 0b1101111;
    let rd = register(pairs.next().unwrap());
    let label = pairs.next().unwrap().as_str();
    let label_position = labels
        .get(label)
        .copied()
        .ok_or(format!("couldn't find label {label}"))?;
    let offset = ((label_position as i32 - position as i32) * 4) as u32;
    let imm = mask_range(20, 1, offset) << 19
        | mask_range(1, 10, offset) << 9
        | mask_range(11, 1, offset) << 8
        | mask_range(12, 8, offset);
    Ok(opcode | (rd << 7) | (imm << 12))
}

fn environment_instruction(immediate: u32) -> Result<u32, String> {
    let opcode: u32 = 0b1110011;
    let funct3: u32 = 0x0;
    let rs: u32 = 0;
    let rd: u32 = 0;
    Ok(opcode | (rd << 7) | (funct3 << 12) | (rs << 15) | (immediate << 20))
}

fn encode_instruction(
    instruction: Pair<Rule>,
    labels: &HashMap<String, u32>,
    position: u32,
) -> Result<u32, String> {
    let rule = instruction.as_rule();
    let pairs = instruction.into_inner();
    match rule {
        Rule::instr_lb => load_instruction(pairs, 0x0),
        Rule::instr_lh => load_instruction(pairs, 0x1),
        Rule::instr_lw => load_instruction(pairs, 0x2),
        Rule::instr_lbu => load_instruction(pairs, 0x4),
        Rule::instr_lhu => load_instruction(pairs, 0x5),
        Rule::instr_addi => immediate_arithmetic_instruction(pairs, 0x0, None),
        Rule::instr_slli => immediate_arithmetic_instruction(pairs, 0x1, Some(0x00)),
        Rule::instr_slti => immediate_arithmetic_instruction(pairs, 0x2, None),
        Rule::instr_sltiu => immediate_arithmetic_instruction(pairs, 0x3, None),
        Rule::instr_xori => immediate_arithmetic_instruction(pairs, 0x4, None),
        Rule::instr_srli => immediate_arithmetic_instruction(pairs, 0x5, Some(0x00)),
        Rule::instr_srai => immediate_arithmetic_instruction(pairs, 0x5, Some(0x20)),
        Rule::instr_ori => immediate_arithmetic_instruction(pairs, 0x6, None),
        Rule::instr_andi => immediate_arithmetic_instruction(pairs, 0x7, None),
        Rule::instr_auipc => lui_or_auipc(pairs, 0b0010111),
        Rule::instr_sb => store_instruction(pairs, 0x0),
        Rule::instr_sh => store_instruction(pairs, 0x1),
        Rule::instr_sw => store_instruction(pairs, 0x2),
        Rule::instr_add => register_arithmetic_instruction(pairs, 0x0, 0x00),
        Rule::instr_sub => register_arithmetic_instruction(pairs, 0x0, 0x20),
        Rule::instr_sll => register_arithmetic_instruction(pairs, 0x1, 0x00),
        Rule::instr_slt => register_arithmetic_instruction(pairs, 0x2, 0x00),
        Rule::instr_sltu => register_arithmetic_instruction(pairs, 0x3, 0x00),
        Rule::instr_xor => register_arithmetic_instruction(pairs, 0x4, 0x00),
        Rule::instr_srl => register_arithmetic_instruction(pairs, 0x5, 0x00),
        Rule::instr_sra => register_arithmetic_instruction(pairs, 0x5, 0x20),
        Rule::instr_or => register_arithmetic_instruction(pairs, 0x6, 0x00),
        Rule::instr_and => register_arithmetic_instruction(pairs, 0x7, 0x00),
        Rule::instr_lui => lui_or_auipc(pairs, 0b0110111),
        Rule::instr_beq => branch_instruction(labels, position, pairs, 0x0),
        Rule::instr_bne => branch_instruction(labels, position, pairs, 0x1),
        Rule::instr_blt => branch_instruction(labels, position, pairs, 0x4),
        Rule::instr_bge => branch_instruction(labels, position, pairs, 0x5),
        Rule::instr_bltu => branch_instruction(labels, position, pairs, 0x6),
        Rule::instr_bgeu => branch_instruction(labels, position, pairs, 0x7),
        Rule::instr_jalr => jalr(pairs),
        Rule::instr_jal => jal(labels, position, pairs),
        Rule::instr_ecall => environment_instruction(0x0),
        Rule::instr_ebreak => environment_instruction(0x1),
        _ => unreachable!(),
    }
}

pub fn assemble(source: &str) -> Result<Vec<u32>, String> {
    let mut position = 0;
    let mut labels = HashMap::new();

    let top_level_pairs = AsmParser::parse(Rule::program, source)
        .map_err(|error| error.to_string())?
        .next()
        .unwrap()
        .into_inner();
    for top_level_pair in top_level_pairs.clone() {
        match top_level_pair.as_rule() {
            Rule::label => {
                let ident = top_level_pair.into_inner().next().unwrap().as_str();
                labels.insert(ident.to_owned(), position);
            }
            Rule::instr => {
                position += 1;
            }
            _ => unreachable!(),
        }
    }

    let mut instructions = Vec::new();
    for top_level_pair in top_level_pairs {
        match top_level_pair.as_rule() {
            Rule::label => {
                // do nothing
            }
            Rule::instr => {
                let pair = top_level_pair.into_inner().next().unwrap();
                let position = instructions.len() as u32;
                let instruction = encode_instruction(pair, &labels, position)?;
                instructions.push(instruction);
            }
            _ => unreachable!(),
        }
    }

    Ok(instructions)
}
