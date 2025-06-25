import itertools
import logging
import time
import random
import os
from multiprocessing import Pool, cpu_count, shared_memory
import pybind_derive2pub
import sys
import numpy as np
from numba import njit
import hashlib


@njit
def indices_to_entropy(indices, checksum_len):
    bit_array = np.zeros(len(indices) * 11, dtype=np.uint8)
    for i in range(len(indices)):
        val = indices[i]
        for j in range(11):
            bit_array[i * 11 + j] = (val >> (10 - j)) & 1

    entropy_len = len(bit_array) - checksum_len
    entropy = np.zeros(entropy_len // 8, dtype=np.uint8)
    for i in range(0, entropy_len, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bit_array[i + j]
        entropy[i // 8] = byte
    return entropy


@njit
def batch_match(pubkeys_array, target_bytes):
    for i in range(pubkeys_array.shape[0]):
        match = True
        for j in range(len(target_bytes)):
            if pubkeys_array[i, j] != target_bytes[j]:
                match = False
                break
        if match:
            return i
    return -1


class MnemonicValidator:
    wordlist = None
    index_map = None

    def __init__(self, wordlist_data=None):
        if MnemonicValidator.wordlist is None and wordlist_data:
            decoded = wordlist_data.decode("utf-8").splitlines()
            MnemonicValidator.wordlist = [line.strip() for line in decoded if line.strip()]
            MnemonicValidator.index_map = {w: i for i, w in enumerate(MnemonicValidator.wordlist)}
        self.wordlist = MnemonicValidator.wordlist
        self.index_map = MnemonicValidator.index_map

    def is_valid(self, mnemonic: str) -> bool:
        words = mnemonic.strip().split()
        if len(words) not in [12, 15, 18, 21, 24] or len(set(words)) != len(words):
            return False
        try:
            indices = [self.index_map[w] for w in words]
        except KeyError:
            return False
        bit_len = len(indices) * 11
        checksum_len = bit_len // 33
        entropy = indices_to_entropy(np.array(indices), checksum_len)
        hash_bytes = hashlib.sha256(entropy).digest()
        hash_bits = bin(int.from_bytes(hash_bytes, "big"))[2:].zfill(256)
        expected_checksum = hash_bits[:checksum_len]
        bit_array = ''.join(f"{i:011b}" for i in indices)
        actual_checksum = bit_array[-checksum_len:]
        return actual_checksum == expected_checksum


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

custom_words = """
fire...
""".split()

path_indices = [44 | 0x80000000, 195 | 0x80000000, 0 | 0x80000000, 0, 0]
passphrase = ""
threads_per_block = 256
batch_size_gpu = 20000
batch_size_cpu = 240000
sub_batch_size = 30000

checkpoint_path = "progress_checkpoint.txt"
target_pubkey_hex = "04..."
target_pubkey_bytes = np.frombuffer(bytes.fromhex(target_pubkey_hex), dtype=np.uint8)


def validate_batch(batch):
    shm = shared_memory.SharedMemory(name="wordlist_shm")
    wordlist_data = bytes(shm.buf[:])
    validator = MnemonicValidator(wordlist_data=wordlist_data)
    return [' '.join(words) for words in batch if validator.is_valid(' '.join(words))]


def batched(it, size):
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def main():
    logger.info("Loading wordlist and preparing shared memory...")
    with open("english.txt", "rb") as f:
        wordlist_data = f.read()
    shm = shared_memory.SharedMemory(create=True, size=len(wordlist_data), name="wordlist_shm")
    shm.buf[:len(wordlist_data)] = wordlist_data
    MnemonicValidator(wordlist_data=wordlist_data)

    logger.info("Generating all 12-word combinations...")
    all_combinations = itertools.combinations(custom_words, 12)

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            start_batch_id = int(f.read().strip())
        logger.info(f"[Resume] Skipping to batch {start_batch_id}...")
        all_combinations = itertools.islice(all_combinations, start_batch_id * batch_size_cpu, None)
    else:
        start_batch_id = 0

    logger.info("Validating mnemonics and deriving pubkeys on-the-fly...")
    pool = Pool(cpu_count() * 2)
    batch_id = start_batch_id
    total_pubkeys = 0

    with open("matched_mnemonics.txt", "w", encoding="utf-8") as fout, \
         open("samples_logged.txt", "w", encoding="utf-8") as sample_log:
        for batch in batched(all_combinations, batch_size_cpu):
            start = time.time()
            sub_batches = [batch[i:i+sub_batch_size] for i in range(0, len(batch), sub_batch_size)]
            results = pool.map(validate_batch, sub_batches)
            valid_batch = [m for group in results for m in group]
            logger.info(f"[CPU] Batch {batch_id}: validated {len(batch)} → {len(valid_batch)} valid in {time.time() - start:.2f}s")

            if valid_batch:
                start_gpu = time.time()
                pubkeys = pybind_derive2pub.derive_pubkeys(
                    mnemonics=valid_batch,
                    passphrase=passphrase,
                    path_indices=path_indices,
                    threads_per_block=threads_per_block
                )
                logger.info(f"[GPU] Derived {len(pubkeys)} pubkeys in {time.time() - start_gpu:.2f}s")

                pubkeys = [bytes(p) for p in pubkeys]
                pubkey_matrix = np.array([np.frombuffer(p, dtype=np.uint8) for p in pubkeys])
                matched_idx = batch_match(pubkey_matrix, target_pubkey_bytes)
                if matched_idx != -1:
                    matched_mnemonic = valid_batch[matched_idx]
                    logger.info(f"[✓] MATCH FOUND! {matched_mnemonic} → {target_pubkey_hex}")
                    fout.write(matched_mnemonic + "\n")
                    fout.flush()
                    pool.terminate()
                    pool.join()
                    shm.close()
                    shm.unlink()
                    sys.exit(0)

                if batch_id % 20 == 0:
                    idx = random.randint(0, len(valid_batch) - 1)
                    sample_log.write(f"{valid_batch[idx]} → {pubkeys[idx].hex()}\n")
                    sample_log.flush()

                total_pubkeys += len(pubkeys)

            if batch_id % 10 == 0:
                with open(checkpoint_path, "w") as f:
                    f.write(str(batch_id))
            batch_id += 1

    pool.close()
    pool.join()
    shm.close()
    shm.unlink()
    logger.info(f"[✓] Total pubkeys derived: {total_pubkeys}")


if __name__ == "__main__":
    main()
