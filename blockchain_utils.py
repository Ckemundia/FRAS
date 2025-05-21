import os
import json
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
PUBLIC_KEY = os.getenv("PUBLIC_KEY")
WEB3_PROVIDER = os.getenv("WEB3_PROVIDER")
NFT_CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("NFT_CONTRACT_ADDRESS"))
TOKEN_CONTRACT_ADDRESS = Web3.to_checksum_address(os.getenv("TOKEN_CONTRACT_ADDRESS"))

assert all([PRIVATE_KEY, PUBLIC_KEY, WEB3_PROVIDER, NFT_CONTRACT_ADDRESS, TOKEN_CONTRACT_ADDRESS]), "❌ Missing environment variables"

# Connect to blockchain
w3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER))
assert w3.is_connected(), "⚠️ Web3 provider connection failed"

# Load ABIs
with open("nft_abi.json") as f:
    nft_abi = json.load(f)

with open("token_abi.json") as f:
    token_abi = json.load(f)

# Initialize contracts
nft_contract = w3.eth.contract(address=NFT_CONTRACT_ADDRESS, abi=nft_abi)
token_contract = w3.eth.contract(address=TOKEN_CONTRACT_ADDRESS, abi=token_abi)
# NFT metadata already uploaded to IPFS — hardcoded token URI from CID
token_uri = "ipfs://bafkreibghqpaxdqyzhjv6tpj7pm63dy6ykxlljfj2spf57zikkx4srv6xq"

# Load NFT metadata from 100Attendance.json
try:
    with open("100Attendance.json", "r") as f:
        nft_metadata = json.load(f)
    token_uri = "ipfs://bafkreibghqpaxdqyzhjv6tpj7pm63dy6ykxlljfj2spf57zikkx4srv6xq"

except Exception as e:
    print(f"❌ Failed to load NFT metadata: {e}")
    token_uri = None  # Safe fallback
    # You can optionally hardcode or fetch from IPFS here

# -----------------------------
# Helper: Send signed transaction
# -----------------------------
def send_transaction(txn):
    try:
        signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Tx successful: {receipt.transactionHash.hex()}")
        return receipt
    except Exception as e:
        print(f"🚨 Transaction failed: {e}")
        return None

# -----------------------------
# Send ERC-20 Token to Student
# -----------------------------
def send_token_to_wallet(wallet, amount=1):
    try:
        student_wallet = Web3.to_checksum_address(wallet)
        decimals = token_contract.functions.decimals().call()
        amount_wei = int(amount * (10 ** decimals))
        nonce = w3.eth.get_transaction_count(PUBLIC_KEY)

        balance = w3.eth.get_balance(PUBLIC_KEY)
        print(f"💰 Sender ETH balance: {w3.from_wei(balance, 'ether')} ETH")

        print(f"🔁 Sending {amount} tokens ({amount_wei} wei) to {wallet}")

        txn = token_contract.functions.transfer(wallet, amount_wei).build_transaction({
            'from': PUBLIC_KEY,
            'nonce': nonce,
            'gas': 250000,
            'gasPrice': w3.to_wei('15', 'gwei')
        })

        return send_transaction(txn)
    except Exception as e:
        print(f"❌ Token transfer failed: {e}")
        return None


# -----------------------------
# Mint NFT for Student
# -----------------------------
def mint_student_nft(wallet, uri_override=None):
    try:
        uri = uri_override if uri_override else token_uri
        if not uri:
            print("⚠️ No token URI available to mint NFT.")
            return None

        nonce = w3.eth.get_transaction_count(PUBLIC_KEY)
        txn = nft_contract.functions.mintNFT(wallet, uri).build_transaction({
            'from': PUBLIC_KEY,
            'nonce': nonce,
            'gas': 300000,
            'gasPrice': w3.to_wei('10', 'gwei')
        })
        return send_transaction(txn)
    except Exception as e:
        print(f"❌ NFT minting failed: {e}")
        return None
