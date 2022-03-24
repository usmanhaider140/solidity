abstract contract A {
    function f(uint256[] calldata a) external virtual returns (uint256);
}

contract B is A {
    function f(uint256[] calldata a) public override returns (uint256) {
        return a[0];
    }

    function g(uint[] calldata x) public returns (uint256) {
        return f(x);
    }
}
// ====
// compileViaYul: also
// ----
// f(uint256[]): 0x20, 1, 9 -> 23
// g(uint256[]): 0x20, 1, 9 -> 23
