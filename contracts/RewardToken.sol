// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract RewardToken is ERC20 {
    address public admin;

    constructor() ERC20("AttendanceCoin", "ATC") {
        admin = msg.sender;
        _mint(msg.sender, 1000000 * 10 ** decimals()); // 1 million tokens
    }

    function reward(address to, uint amount) external {
        require(msg.sender == admin, "Only admin can reward");
        _transfer(admin, to, amount);
    }
}
