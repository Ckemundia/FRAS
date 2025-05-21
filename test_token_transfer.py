import os
import json
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
PRIVATE_KEY = os.getenv("PRIVATE_KEY").replace('"', '')  # Strip quotes if present
PUBLIC_KEY = os.getenv("PUBLIC_KEY").replace('"', '')
WEB3_PROVIDER = os.getenv("WEB3_PROVIDER").replace('"', '')
TOKEN_CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("TOKEN_CONTRACT_ADDRESS").replace('"', ''))

# Connect to Web3
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
assert w3.is_connected(), "❌ Web3 connection failed"

# Load token ABI
with open("token_abi.json") as f:
    token_abi = json.load(f)

# Init token contract
token_contract = w3.eth.contract(address=TOKEN_CONTRACT_ADDRESS, abi=token_abi)

# Test recipient (can be any valid wallet you control)
TO_ADDRESS = Web3.to_checksum_address("0xF63e687a7619833c6dB3B43763194222464a4D11")  # Replace or keep same

def send_test_token(to_address, amount=1):
    try:
        decimals = token_contract.functions.decimals().call()
        amount_wei = int(amount * (10 ** decimals))
        nonce = w3.eth.get_transaction_count(PUBLIC_KEY)

        print(f"📦 Sending {amount} tokens ({amount_wei} wei) to {to_address}")
        print(f"🔢 Nonce: {nonce}")
        print(f"💰 Sender ETH Balance: {w3.from_wei(w3.eth.get_balance(PUBLIC_KEY), 'ether')} ETH")

        txn = token_contract.functions.transfer(to_address, amount_wei).build_transaction({
            'from': PUBLIC_KEY,
            'nonce': nonce,
            'gas': 250000,
            'gasPrice': w3.to_wei('15', 'gwei')
        })

        signed_txn = w3.eth.account.sign_transaction(txn, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

        print(f"🚀 Transaction sent! Tx hash: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Transaction confirmed in block {receipt.blockNumber}")
        return receipt
    except Exception as e:
        print(f"❌ Error sending token: {e}")
        return None

# Run test
send_test_token(TO_ADDRESS, 1)
