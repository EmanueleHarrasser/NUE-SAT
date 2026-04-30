#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(pub u32);

impl Var {
    pub fn new(index: u32) -> Self {
        assert!(index > 0);
        Var(index)
    }

    pub fn index(self) -> usize {
        (self.0 - 1) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Lit {
    pub var: Var,
    pub neg: bool,
}

impl Lit {
    pub fn new(var: Var, neg: bool) -> Self {
        Lit { var, neg }
    }

    pub fn from_dimacs(value: i32) -> Self {
        assert!(value != 0);
        let neg = value < 0;
        let var = Var::new(value.abs() as u32);
        Lit { var, neg }
    }

    pub fn dimacs(self) -> i32 {
        let v = self.var.0 as i32;
        if self.neg {
            -v
        } else {
            v
        }
    }
}

pub type Clause = Vec<Lit>;

#[derive(Clone, Debug)]
pub struct Cnf {
    pub num_vars: u32,
    pub clauses: Vec<Clause>,
}

impl Cnf {
    pub fn new(num_vars: u32, clauses: Vec<Clause>) -> Self {
        Cnf { num_vars, clauses }
    }
}
