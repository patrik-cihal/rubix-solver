use std::{cmp::Ordering, collections::{HashMap, HashSet}, ops::{Index, Mul}};
use colored::*;

const RSZ: usize = 54;

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq)]
enum Color {
    #[default]
    White,
    Green,
    Yellow,
    Blue,
    Orange,
    Red
}

impl Ord for Color {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}


/// U, F, D, B, L, R
type Move = [usize; RSZ];

const fn mul(mv1: Move, mv2: Move) -> Move {
    let mut res = [0; RSZ];
    let mut i = 0;
    while i < 54 {
        res[i] = mv2[mv1[i]];
        i += 1;
    }
    res
}

const fn inv(mv: Move) -> Move {
    let mut res = [0; RSZ];
    let mut i = 0;
    while i<54 {
        res[mv[i]] = i;
        i += 1;
    } 
    res
}

const ID_MOVE: Move = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 
    9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53
];

/// 6 -> 45; 7 -> 48; 8 -> 51; 51 -> 18; 48 -> 19; 45 -> 20; 18 -> 38; 19 -> 41; 20 -> 44; 44 -> 6; 41 -> 7; 38 -> 8; 15 -> 9; 12 -> 10; 9 -> 11; 10 -> 14; 11 -> 17 14 -> 16; 17 -> 15 (16 -> 12
const F_MOVE: Move = [
    0, 1, 2, 3, 4, 5, 45, 48, 51,
    11, 14, 17, 10, 13, 16, 9, 12, 15,
    38, 41, 44, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 8, 39, 40, 7, 42, 43, 6,
    20, 46, 47, 19, 49, 50, 18, 52, 53
];

const X_MOVE: Move = [
    27, 28, 29, 30, 31, 32, 33, 34, 35,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 
    9, 10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 26,
    42, 39, 36, 43, 40, 37, 44, 41, 38,
    47, 50, 53, 46, 49, 52, 45, 48, 51
];

const Y_MOVE: Move = [
    2, 5, 8, 1, 4, 7, 0, 3, 6,
    36, 37, 38, 39, 40, 41, 42, 43, 44,
    24, 21, 18, 25, 22, 19, 26, 23, 20,
    53, 52, 51, 50, 49, 48, 47, 46, 45,
    35, 34, 33, 32, 31, 30, 29, 28, 27,
    9, 10, 11, 12, 13, 14, 15, 16, 17,
];

macro_rules! mul_seq {
    // Base case: Only one element remains after all multiplications, which is the final result.
    ($x:expr) => { $x };

    // Recursive case: Two elements, just multiply them.
    ($a:expr, $b:expr) => {
        mul($a, $b)
    };

    // General recursive case: More than two elements.
    ($a:expr, $b:expr, $($rest:expr),+ $(,)?) => {
        mul_seq!(mul($a, $b), $($rest),+)
    };
}

const XC_MOVE: Move = inv(X_MOVE);
const YC_MOVE: Move = inv(Y_MOVE);
const R_MOVE: Move = mul_seq!(Y_MOVE, F_MOVE, YC_MOVE);
const L_MOVE: Move = mul_seq!(YC_MOVE, F_MOVE, Y_MOVE);
const B_MOVE: Move = mul_seq!(Y_MOVE, Y_MOVE, F_MOVE, Y_MOVE, Y_MOVE);
const U_MOVE: Move = mul_seq!(XC_MOVE, F_MOVE, X_MOVE);
const D_MOVE: Move = mul_seq!(X_MOVE, F_MOVE, XC_MOVE);
const FC_MOVE: Move = inv(F_MOVE);
const RC_MOVE: Move = inv(R_MOVE);
const LC_MOVE: Move = inv(L_MOVE);
const BC_MOVE: Move = inv(B_MOVE);
const UC_MOVE: Move = inv(U_MOVE);
const DC_MOVE: Move = inv(D_MOVE);


fn apply_move<T: Default+Clone+Copy>(data: &[T; RSZ], mv: Move) -> [T; RSZ] {
    let mut new_data = [T::default(); RSZ];
    for i in 0..RSZ {
        new_data[mv[i]] = data[i];
    }
    new_data
}

type Rubix = [Color; RSZ];
type PartialRubix = [Option<Color>; RSZ];

fn to_colored_color(c: Color) -> colored::Color {
    match c {
        Color::White => colored::Color::White,
        Color::Green => colored::Color::Green,
        Color::Yellow => colored::Color::Yellow,
        Color::Blue => colored::Color::Blue,
        Color::Orange => colored::Color::BrightMagenta,
        Color::Red => colored::Color::Red,
    }
}

fn display_rubix(r: Rubix) {
    for i in 0..6 {
        for j in 0..3 {
            for k in 0..3 {
                print!("{}", "  ".on_color(to_colored_color(r[i*9+j*3+k])))
            }
            println!();
        }
        println!();
    }
}

static mut HASH: [u64; RSZ] = [0; RSZ];
const MOD: u64 = 12905128903561239083;

fn hash(r: &Rubix) -> u64 {
    let mut res = 0;
    for i in 0..RSZ {
        let c = r[i] as u32;
        let mut hsh = unsafe {HASH[i]} as u128;
        for _ in 0..c+1 {
            hsh *= unsafe {HASH[i]} as u128;
            hsh %= MOD as u128;
        } 
        res ^= hsh as u64;
    }
    res
}

const SOLVED_RUBIX: Rubix = [Color::White, Color::White, Color::White, Color::White, Color::White, Color::White, Color::White, Color::White, Color::White,
    Color::Green, Color::Green, Color::Green, Color::Green, Color::Green, Color::Green, Color::Green, Color::Green, Color::Green,
    Color::Yellow, Color::Yellow, Color::Yellow, Color::Yellow, Color::Yellow, Color::Yellow, Color::Yellow, Color::Yellow, Color::Yellow,
    Color::Blue, Color::Blue, Color::Blue, Color::Blue, Color::Blue, Color::Blue, Color::Blue, Color::Blue, Color::Blue,
    Color::Orange, Color::Orange, Color::Orange, Color::Orange, Color::Orange, Color::Orange, Color::Orange, Color::Orange, Color::Orange,
    Color::Red, Color::Red, Color::Red, Color::Red, Color::Red, Color::Red, Color::Red, Color::Red, Color::Red];

const CROSS: PartialRubix = [
    None, Some(Color::White), None, Some(Color::White), Some(Color::White), Some(Color::White), None, Some(Color::White), None,
    None, Some(Color::Green), None, None, Some(Color::Green), None, None, None, None,
    None, None, None, None, None, None, None, None, None,
    None, None, None, None, Some(Color::Blue), None, None, Some(Color::Blue), None,
    None, Some(Color::Orange), None, None, Some(Color::Orange), None, None, None, None,
    None, Some(Color::Red), None, None, Some(Color::Red), None, None, None, None,
];

const SECOND_LAYER_GREEN: PartialRubix = [
    None, Some(Color::White), None, Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), None,
    Some(Color::Green), Some(Color::Green), None, Some(Color::Green), Some(Color::Green), None, None, None, None,
    None, None, None, None, None, None, None, None, None,
    None, None, None, None, Some(Color::Blue), None, None, Some(Color::Blue), None,
    None, Some(Color::Orange), Some(Color::Orange), None, Some(Color::Orange), Some(Color::Orange), None, None, None,
    None, Some(Color::Red), None, None, Some(Color::Red), None, None, None, None,
];

const SECOND_LAYER_ORANGE: PartialRubix = [
    Some(Color::White), Some(Color::White), None, Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), None,
    Some(Color::Green), Some(Color::Green), None, Some(Color::Green), Some(Color::Green), None, None, None, None,
    None, None, None, None, None, None, None, None, None,
    None, None, None, Some(Color::Blue), Some(Color::Blue), None, Some(Color::Blue), Some(Color::Blue), None,
    Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), None, Some(Color::Orange), Some(Color::Orange), None, None, None,
    None, Some(Color::Red), None, None, Some(Color::Red), None, None, None, None,
];

const SECOND_LAYER_RED: PartialRubix = [
    Some(Color::White), Some(Color::White), None, Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White),
    Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), None, None, None,
    None, None, None, None, None, None, None, None, None,
    None, None, None, Some(Color::Blue), Some(Color::Blue), None, Some(Color::Blue), Some(Color::Blue), None,
    Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), None, Some(Color::Orange), Some(Color::Orange), None, None, None,
    Some(Color::Red), Some(Color::Red), None, Some(Color::Red), Some(Color::Red), None, None, None, None,
];

const SECOND_LAYER: PartialRubix = [
    Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White),
    Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), None, None, None,
    None, None, None, None, None, None, None, None, None,
    None, None, None, Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue),
    Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), None, None, None,
    Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), None, None, None,
];

const YELLOW_CROSS: PartialRubix = [
    Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White),
    Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), None, None, None,
    None, Some(Color::Yellow), None, Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), None, Some(Color::Yellow), None,
    None, None, None, Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue),
    Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), None, None, None,
    Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), None, None, None,
];

const YELLOW_FULL: PartialRubix = [
    Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White),
    Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), None, None, None,
    Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow),
    None, None, None, Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue),
    Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), None, None, None,
    Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), None, None, None,
];


const FULL_SOLVE: PartialRubix = [
    Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White), Some(Color::White),
    Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green), Some(Color::Green),
    Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow), Some(Color::Yellow),
    Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue), Some(Color::Blue),
    Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange), Some(Color::Orange),
    Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red), Some(Color::Red),
];

fn cmp_rubix(r: &Rubix, pr: &PartialRubix) -> Ordering {
    for i in 0..RSZ {
        let Some(pri) = pr[i] else {
            continue;
        };
        if r[i] > pri {
            return Ordering::Greater;
        }
        else if r[i] < pri {
            return Ordering::Less;
        }

    }
    return Ordering::Equal;
}

#[derive(Clone, Copy, Debug)]
enum Turn {
    U, F, D, B, L, R, X, Y,
    UC, FC, DC, BC, LC, RC, XC, YC,
    ID
}

impl Into<Move> for Turn {
    fn into(self) -> Move {
        match self {
            Turn::U => U_MOVE,
            Turn::F => F_MOVE,
            Turn::D => D_MOVE,
            Turn::B => B_MOVE,
            Turn::L => L_MOVE,
            Turn::R => R_MOVE,
            Turn::X => X_MOVE,
            Turn::Y => Y_MOVE,
            Turn::UC => UC_MOVE,
            Turn::FC => FC_MOVE,
            Turn::DC => DC_MOVE,
            Turn::BC => BC_MOVE,
            Turn::LC => LC_MOVE,
            Turn::RC => RC_MOVE,
            Turn::XC => XC_MOVE,
            Turn::YC => YC_MOVE,
            Turn::ID => ID_MOVE
        }
    }
}

const SEARCH_TURNS: [Turn; 12] = [Turn::F, Turn::L, Turn::R, Turn::B, Turn::D, Turn::U, Turn::FC, Turn::LC, Turn::RC, Turn::BC, Turn::DC, Turn::UC];

fn find_common(v1: &Vec<Rubix>, v2: &Vec<PartialRubix>) -> Option<Rubix> {
    for i in 0..v1.len() {
        for j in 0..v2.len() {
            if cmp_rubix(&v1[i], &v2[j]).is_eq() {
                return Some(v1[i]);
            }
        }
    }
    return None;
}

fn find_common_full(v1: &Vec<Rubix>, v2: &Vec<Rubix>) -> Option<Rubix> {
    let mut hshs = HashSet::new();
    let mut found = None;
    for r in v2 {
        hshs.insert(hash(r));
    }
    for r in v1 {
        if hshs.contains(&hash(r)) {
            found = Some(hash(r));
            break;
        }
    }
    let Some(found) = found else {
        return None;
    };

    for r in v1 {
        if hash(r) == found {
            return Some(*r);
        }
    }
    panic!("Shouldn't get here!")
}

fn full_bfs_sqrt(r: Rubix) -> Vec<Turn> {
    let mut f_statess = vec![vec![(Turn::ID, r)]];
    let mut b_statess = vec![vec![(Turn::ID, SOLVED_RUBIX)]];

    let found;

    let mut i = 0;
    loop {
        let mut f_cur = f_statess.last().unwrap().iter().map(|x| x.1).collect::<Vec<_>>();
        let mut b_cur = b_statess.last().unwrap().iter().map(|x| x.1).collect::<Vec<_>>();

        if let Some(c) = find_common_full(&f_cur, &b_cur) {
            found = c;
            break;
        }

        if i%2 == 0 {
            let mut f_nxt_states = vec![];
            for state in f_cur {
                for turn in SEARCH_TURNS {
                    f_nxt_states.push((turn, apply_move(&state, turn.into())))
                }
            }
            f_statess.push(f_nxt_states);
        }
        else {
            let mut b_nxt_states = vec![];
            for state in b_cur {
                for turn in SEARCH_TURNS {
                    let mv = inv(turn.into());
                    b_nxt_states.push((turn, apply_move(&state, mv)))
                }
            }
            b_statess.push(b_nxt_states);

        }
        i += 1;
        println!("{i}");
    }

    let mut cur_r = found; 
    let mut turns = vec![];
    while f_statess.len() > 1 {
        for (turn, state) in f_statess.pop().unwrap() {
            if state == cur_r {
                turns.push(turn);
                cur_r = apply_move(&cur_r, inv(turn.into()));
                break;
            }
        }
    }
    turns.reverse();
    let mut cur_r = found;
    while b_statess.len() > 1 {
        for (turn, state) in b_statess.pop().unwrap() {
            if state == cur_r {
                turns.push(turn);
                cur_r = apply_move(&cur_r, turn.into());
                break;
            }
        }
    }

    turns
}

fn bfs_sqrt(r: Rubix, tr: PartialRubix) -> Vec<Turn> {
    let mut f_statess = vec![vec![(Turn::ID, r)]];
    let mut b_statess = vec![vec![(Turn::ID, tr)]];

    let found;

    let mut i = 0;
    loop {
        let mut f_cur = f_statess.last().unwrap().iter().map(|x| x.1).collect::<Vec<_>>();
        let mut b_cur = b_statess.last().unwrap().iter().map(|x| x.1).collect::<Vec<_>>();

        if let Some(c) = find_common(&f_cur, &b_cur) {
            found = c;
            break;
        }

        if i%2 == 0 {
            let mut f_nxt_states = vec![];
            for state in f_cur {
                for turn in SEARCH_TURNS {
                    f_nxt_states.push((turn, apply_move(&state, turn.into())))
                }
            }
            f_statess.push(f_nxt_states);
        }
        else {
            let mut b_nxt_states = vec![];
            for state in b_cur {
                for turn in SEARCH_TURNS {
                    let mv = inv(turn.into());
                    b_nxt_states.push((turn, apply_move(&state, mv)))
                }
            }
            b_statess.push(b_nxt_states);

        }
        i += 1;
        println!("{i}");
    }

    let mut cur_r = found; 
    let mut turns = vec![];
    while f_statess.len() > 1 {
        for (turn, state) in f_statess.pop().unwrap() {
            if state == cur_r {
                turns.push(turn);
                cur_r = apply_move(&cur_r, inv(turn.into()));
                break;
            }
        }
    }
    turns.reverse();
    let mut cur_r = found;
    while b_statess.len() > 1 {
        for (turn, state) in b_statess.pop().unwrap() {
            if cmp_rubix(&cur_r, &state).is_eq() {
                turns.push(turn);
                cur_r = apply_move(&cur_r, turn.into());
                break;
            }
        }
    }

    turns
}

fn bfs(r: Rubix, tr: PartialRubix) -> Vec<Turn> {
    let mut statess = vec![vec![(Turn::ID, r)]];
    let found;
    'outer: loop {
        let cur_states = statess.last().unwrap();
        for (turn, state) in cur_states {
            if cmp_rubix(state, &tr).is_eq() {
                found = (*turn, *state);
                break 'outer;
            }
        }
        let mut nxt_states = vec![];
        for (_, state) in cur_states {
            for turn in SEARCH_TURNS {
                nxt_states.push((turn, apply_move(state, turn.into())))
            }
        }
        statess.push(nxt_states);
    }

    let mut cur_trans = found;
    let mut turns = vec![];
    while statess.len() > 1 {
        statess.pop();
        let prev_r = apply_move(&cur_trans.1, inv(cur_trans.0.into()));
        turns.push(cur_trans.0);
        for (turn, state) in statess.last().unwrap() {
            if state == &prev_r {
                cur_trans = (*turn, *state);
                break;
            }
        }
    }
    turns.reverse();
    turns
}

fn apply_moves(mut r: Rubix, mvs: Vec<Move>) -> Rubix {
    for mv in mvs {
        r = apply_move(&r, mv);
    } 
    r
}

fn full_solve(mut r: Rubix, steps: Vec<PartialRubix>) -> Vec<Turn> {
    let mut result = vec![];
    for (i, pr) in steps.into_iter().enumerate() {
        let mut mvs = bfs_sqrt(r, pr);
        r = apply_moves(r, mvs.iter().map(|&x| x.into()).collect::<Vec<Move>>());
        result.append(&mut mvs);

        println!("Phase {i} finished!");
    }
    result
}

fn main() {
    for i in 0..RSZ {
        unsafe { HASH[i] = rand::random::<u64>() }
    }

    let mixed_cube = apply_move(&SOLVED_RUBIX, mul_seq!(FC_MOVE, R_MOVE, L_MOVE, U_MOVE, R_MOVE, D_MOVE, L_MOVE, U_MOVE, F_MOVE, U_MOVE, R_MOVE, L_MOVE, F_MOVE));
    display_rubix(mixed_cube);

    //println!("{:?}", full_solve(mixed_cube, vec![CROSS, SECOND_LAYER_GREEN, SECOND_LAYER_ORANGE, SECOND_LAYER_RED, SECOND_LAYER, YELLOW_CROSS, YELLOW_FULL]));
    println!("{:?}", full_bfs_sqrt(mixed_cube));
}
