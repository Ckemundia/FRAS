import os
import json
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
PRIVATE_KEY = os.getenv("PRIVATE_KEY").replace('"', '')
PUBLIC_KEY = os.getenv("PUBLIC_KEY").replace('"', '')
WEB3_PROVIDER = os.getenv("WEB3_PROVIDER").replace('"', '')
NFT_CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("NFT_CONTRACT_ADDRESS").replace('"', ''))

# Connect to Web3
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
assert w3.is_connected(), "❌ Web3 connection failed"

# Load ABI
with open("nft_abi.json") as f:
    nft_abi = json.load(f)

# Init contract
nft_contract = w3.eth.contract(address=NFT_CONTRACT_ADDRESS, abi=nft_abi)

# Target recipient wallet (can be your PUBLIC_KEY)
TO_ADDRESS = Web3.to_checksum_address("0xF63e687a7619833c6dB3B43763194222464a4D11")

# Example metadata URI (ensure this is pinned on IPFS!)
TOKEN_URI = "ipfs://bafkreibghqpaxdqyzhjv6tpj7pm63dy6ykxlljfj2spf57zikkx4srv6xq"

def mint_nft(to_address, token_uri):
    try:
        nonce = w3.eth.get_transaction_count(PUBLIC_KEY)
        print(f"🎨 Minting NFT to: {to_address}")
        print(f"🔗 Using metadata URI: {token_uri}")
        print(f"🔢 Nonce: {nonce}")
        print(f"💰 ETH Balance: {w3.from_wei(w3.eth.get_balance(PUBLIC_KEY), 'ether')} ETH")

        txn = nft_contract.functions.mintNFT(to_address, token_uri).build_transaction({
            'from': PUBLIC_KEY,
            'nonce': nonce,
            'gas': 300000,
            'gasPrice': w3.to_wei('15', 'gwei')
        })

        signed_txn = w3.eth.account.sign_transaction(txn, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

        print(f"🚀 Tx sent: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ NFT Minted! Confirmed in block {receipt.blockNumber}")
        return receipt
    except Exception as e:
        print(f"❌ NFT minting failed: {e}")
        return None

# Run mint
mint_nft(TO_ADDRESS, TOKEN_URI)
