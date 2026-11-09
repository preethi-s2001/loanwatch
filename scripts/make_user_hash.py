"""
Helper utility to generate secure PBKDF2-HMAC-SHA256 password hashes
for LoanWatch Streamlit secrets configuration (.streamlit/secrets.toml).
Prompts for password securely without echoing to terminal.
"""

import getpass
import hashlib
import secrets
import sys


def generate_hash(iterations: int = 200000) -> str:
    """Prompts for a password and computes its PBKDF2-HMAC-SHA256 hash."""
    pwd = getpass.getpass("Enter password: ")
    if not pwd:
        print("Error: Password cannot be empty.", file=sys.stderr)
        sys.exit(1)

    confirm = getpass.getpass("Confirm password: ")
    if pwd != confirm:
        print("Error: Passwords do not match.", file=sys.stderr)
        sys.exit(1)

    salt = secrets.token_bytes(16)
    hash_bytes = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${hash_bytes.hex()}"


def main():
    try:
        h = generate_hash()
        print("\nGenerated hash (copy into .streamlit/secrets.toml under password_hash):")
        print(h)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
