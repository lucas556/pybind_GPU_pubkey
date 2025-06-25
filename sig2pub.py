import hashlib
import json
import requests
import base58
from eth_keys.datatypes import Signature
from Crypto.Hash import keccak

TRONGRID_API = "https://api.trongrid.io/wallet/gettransactionbyid"
txid = "3e......."

def keccak256(data: bytes) -> bytes:
    """计算Keccak-256哈希"""
    hasher = keccak.new(digest_bits=256)
    hasher.update(data)
    return hasher.digest()

def recover_tron_address(raw_data_hex: str, signature_hex: str) -> str:
    """从交易原始数据和签名恢复Tron地址"""
    # 计算SHA256哈希
    raw_data_bytes = bytes.fromhex(raw_data_hex)
    message_hash = hashlib.sha256(raw_data_bytes).digest()

    # 解析签名
    signature_bytes = bytes.fromhex(signature_hex)
    r = int.from_bytes(signature_bytes[0:32], byteorder='big')
    s = int.from_bytes(signature_bytes[32:64], byteorder='big')
    v = signature_bytes[64]
    if v >= 27:
        v = v - 27

    sig = Signature(vrs=(v, r, s))

    # 恢复公钥
    public_key_bytes = sig.recover_public_key_from_msg_hash(message_hash).to_bytes()
    print("Recovered Public Key (hex):", public_key_bytes.hex())
    # 生成Tron地址
    primitive_addr = b'\x41' + keccak256(public_key_bytes)[-20:]
    tron_address = base58.b58encode_check(primitive_addr)

    return tron_address.decode()

def fetch_transaction(txid: str) -> dict:
    """通过TronGrid API请求交易数据"""
    headers = {"Content-Type": "application/json"}
    payload = {"value": txid}
    response = requests.post(TRONGRID_API, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def main():
    # 拉取交易数据
    tx_data = fetch_transaction(txid)

    # 提取 raw_data 和 signature
    raw_data = tx_data.get('raw_data_hex')
    signatures = tx_data.get('signature')

    if not raw_data or not signatures:
        print("交易数据不完整，无法恢复签名地址。")
        return

    signature = signatures[0]  # 通常只有一个签名

    # 恢复地址
    recovered_address = recover_tron_address(raw_data, signature)
    print("Recovered Tron Address:", recovered_address)

if __name__ == "__main__":
    main()
