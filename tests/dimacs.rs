use enue_sat::cnf::Lit;
use enue_sat::parser::dimacs::parse_dimacs;

#[test]
fn parses_basic_dimacs() {
    let input = "c example\np cnf 3 2\n1 -3 0\n2 3 -1 0\n";
    let cnf = parse_dimacs(input).expect("parse");

    assert_eq!(cnf.num_vars, 3);
    assert_eq!(cnf.clauses.len(), 2);
    assert_eq!(cnf.clauses[0].len(), 2);
    assert_eq!(cnf.clauses[1].len(), 3);
    assert_eq!(cnf.clauses[0][0], Lit::from_dimacs(1));
    assert_eq!(cnf.clauses[0][1], Lit::from_dimacs(-3));
}
