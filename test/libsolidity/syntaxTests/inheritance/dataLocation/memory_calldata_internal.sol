abstract contract A {
    function f(uint256[1] memory a) internal virtual returns (uint256);
    function test() external returns (uint) {
        uint[1] memory t;
        t[0] = 7;
        return f(t);
    }
}

contract B is A {
    function f(uint256[1] calldata a) internal override returns (uint256) {
        return a[0];
    }
}
// ----
// TypeError 7792: (279-287): Function has override specified but does not override anything.
// TypeError 3656: (214-336): Contract "B" should be marked as abstract.
