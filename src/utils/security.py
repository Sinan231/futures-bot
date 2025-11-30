"""
Security utilities for the trading signal system.

Provides HMAC authentication, JWT token management, data encryption,
and secure configuration handling for API credentials and sensitive data.
"""

import base64
import hashlib
import hmac
import json
import jwt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..config import get_config


class SecurityManager:
    """Advanced security manager for trading system."""

    def __init__(self, config: Optional[Any] = None):
        """Initialize security manager with configuration."""
        self.config = config or get_config()
        self._fernet_cipher: Optional[Fernet] = None
        self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialize encryption cipher with PBKDF2 key derivation."""
        if self.config.hmac_secret_key:
            # Derive encryption key from HMAC secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'trading_system_salt',  # In production, use unique salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.config.hmac_secret_key.encode()))
            self._fernet_cipher = Fernet(key)
        else:
            raise ValueError("HMAC secret key required for encryption")

    def generate_hmac_signature(self, payload: Union[str, Dict[str, Any]]) -> str:
        """Generate HMAC signature for payload authentication."""
        if isinstance(payload, dict):
            payload = json.dumps(payload, sort_keys=True, separators=(',', ':'))

        signature = hmac.new(
            self.config.hmac_secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def verify_hmac_signature(
        self,
        payload: Union[str, Dict[str, Any]],
        signature: str
    ) -> bool:
        """Verify HMAC signature for payload."""
        try:
            expected_signature = self.generate_hmac_signature(payload)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False

    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data using Fernet symmetric encryption."""
        if self._fernet_cipher is None:
            raise ValueError("Encryption not initialized")

        if isinstance(data, dict):
            data = json.dumps(data)

        # Convert to bytes if needed
        if isinstance(data, str):
            data = data.encode()

        encrypted_data = self._fernet_cipher.encrypt(data)
        return base64.b64encode(encrypted_data).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data using Fernet symmetric encryption."""
        if self._fernet_cipher is None:
            raise ValueError("Encryption not initialized")

        try:
            # Decode base64 and decrypt
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted_data = self._fernet_cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {e}")

    def generate_jwt_token(
        self,
        payload: Dict[str, Any],
        expires_in_hours: int = 24
    ) -> str:
        """Generate JWT token with expiration."""
        expiration = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

        jwt_payload = {
            **payload,
            "exp": expiration,
            "iat": datetime.now(timezone.utc),
            "iss": "trading_signal_system",
            "ver": "1.0.0"
        }

        return jwt.encode(
            jwt_payload,
            self.config.jwt_secret_key,
            algorithm="HS256"
        )

    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=["HS256"],
                issuer="trading_signal_system",
                options={"require": ["exp", "iat", "iss"]}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("JWT token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid JWT token: {e}")

    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and secret pair."""
        api_key = secrets.token_urlsafe(32)
        api_secret = secrets.token_urlsafe(64)
        return api_key, api_secret

    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash password using PBKDF2."""
        if salt is None:
            salt = secrets.token_hex(16)

        password_hash = hashlib.pbkdf2_hex(
            password.encode(),
            salt.encode(),
            100000,
            dklen=32
        )

        return password_hash, salt

    def verify_password(self, password: str, hash_value: str, salt: str) -> bool:
        """Verify password against hash."""
        computed_hash, _ = self.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, hash_value)

    def generate_nonce(self, length: int = 16) -> str:
        """Generate cryptographically secure nonce."""
        return secrets.token_hex(length)

    def mask_sensitive_info(self, data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
        """Mask sensitive information for logging."""
        if len(data) <= visible_chars:
            return mask_char * len(data)

        return data[:visible_chars] + mask_char * (len(data) - visible_chars)

    def sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data for logging."""
        sensitive_fields = {
            'api_key', 'api_secret', 'secret', 'password', 'token',
            'hmac_signature', 'jwt_token', 'encryption_key', 'salt'
        }

        sanitized_data = {}
        for key, value in data.items():
            if any(sensitive_field in key.lower() for sensitive_field in sensitive_fields):
                if isinstance(value, str):
                    sanitized_data[key] = self.mask_sensitive_info(value)
                else:
                    sanitized_data[key] = "[REDACTED]"
            else:
                sanitized_data[key] = value

        return sanitized_data


class SignalAuthenticator:
    """Authentication and validation for trading signals."""

    def __init__(self, security_manager: Optional[SecurityManager] = None):
        """Initialize signal authenticator."""
        self.security_manager = security_manager or SecurityManager()

    def sign_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Add HMAC signature to trading signal."""
        # Remove any existing signature
        signal_copy = {k: v for k, v in signal.items() if k != 'signature'}

        # Generate signature
        signature = self.security_manager.generate_hmac_signature(signal_copy)

        # Add signature to signal
        signal_copy['signature'] = signature

        return signal_copy

    def verify_signal(self, signal: Dict[str, Any]) -> bool:
        """Verify trading signal signature."""
        if 'signature' not in signal:
            return False

        # Extract signature
        signature = signal['signature']
        signal_copy = {k: v for k, v in signal.items() if k != 'signature'}

        # Verify signature
        return self.security_manager.verify_hmac_signature(signal_copy, signature)

    def validate_signal_structure(self, signal: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate signal structure and required fields."""
        errors = []

        required_fields = [
            'timestamp', 'model_version', 'pair', 'side', 'entry_price',
            'leverage', 'size_percent_of_equity', 'stop_loss', 'tp1',
            'confidence', 'reason', 'signature'
        ]

        optional_fields = [
            'tp2', 'tp3', 'backtest_stats'
        ]

        # Check required fields
        for field in required_fields:
            if field not in signal:
                errors.append(f"Missing required field: {field}")

        # Validate field types and values
        if 'timestamp' in signal:
            try:
                # Should be ISO format string
                datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                errors.append("Invalid timestamp format, should be ISO 8601")

        if 'side' in signal and signal['side'] not in ['long', 'short']:
            errors.append("Side must be 'long' or 'short'")

        if 'confidence' in signal:
            try:
                confidence = float(signal['confidence'])
                if not 0 <= confidence <= 1:
                    errors.append("Confidence must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("Confidence must be a number between 0 and 1")

        if 'leverage' in signal:
            try:
                leverage = float(signal['leverage'])
                if leverage <= 0:
                    errors.append("Leverage must be greater than 0")
            except (ValueError, TypeError):
                errors.append("Leverage must be a positive number")

        price_fields = ['entry_price', 'stop_loss', 'tp1', 'tp2', 'tp3']
        for field in price_fields:
            if field in signal and signal[field] is not None:
                try:
                    price = float(signal[field])
                    if price <= 0:
                        errors.append(f"{field} must be greater than 0")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a positive number")

        if 'size_percent_of_equity' in signal:
            try:
                size = float(signal['size_percent_of_equity'])
                if not 0 < size <= 100:
                    errors.append("Size percent must be between 0 and 100")
            except (ValueError, TypeError):
                errors.append("Size percent must be a number between 0 and 100")

        return len(errors) == 0, errors

    def clean_signal_data(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize signal data."""
        cleaned_signal = {}

        # Copy and normalize fields
        for key, value in signal.items():
            if value is not None:
                if key in ['entry_price', 'stop_loss', 'tp1', 'tp2', 'tp3']:
                    try:
                        cleaned_signal[key] = float(value)
                    except (ValueError, TypeError):
                        cleaned_signal[key] = value
                elif key in ['leverage', 'confidence', 'size_percent_of_equity']:
                    try:
                        cleaned_signal[key] = float(value)
                    except (ValueError, TypeError):
                        cleaned_signal[key] = value
                elif key == 'timestamp':
                    # Ensure ISO format
                    if isinstance(value, str):
                        cleaned_signal[key] = value
                    else:
                        cleaned_signal[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                else:
                    cleaned_signal[key] = value

        return cleaned_signal


class APIKeyManager:
    """Secure API key management and rotation."""

    def __init__(self, security_manager: Optional[SecurityManager] = None):
        """Initialize API key manager."""
        self.security_manager = security_manager or SecurityManager()

    def encrypt_api_credentials(
        self,
        api_key: str,
        api_secret: str
    ) -> Dict[str, str]:
        """Encrypt API credentials for storage."""
        credentials = {
            'api_key': api_key,
            'api_secret': api_secret,
            'encrypted_at': datetime.now(timezone.utc).isoformat()
        }

        encrypted_credentials = self.security_manager.encrypt_sensitive_data(credentials)

        return {
            'encrypted_credentials': encrypted_credentials,
            'checksum': hashlib.sha256(encrypted_credentials.encode()).hexdigest()
        }

    def decrypt_api_credentials(self, encrypted_data: str, checksum: str) -> Optional[Dict[str, str]]:
        """Decrypt API credentials from storage."""
        try:
            # Verify checksum
            computed_checksum = hashlib.sha256(encrypted_data.encode()).hexdigest()
            if not secrets.compare_digest(computed_checksum, checksum):
                raise ValueError("Checksum verification failed")

            # Decrypt credentials
            decrypted_data = self.security_manager.decrypt_sensitive_data(encrypted_data)
            credentials = json.loads(decrypted_data)

            return credentials

        except Exception as e:
            print(f"Failed to decrypt API credentials: {e}")
            return None

    def rotate_api_key(self, current_encrypted_data: str) -> Dict[str, str]:
        """Rotate API key pair."""
        # Decrypt current credentials
        current_checksum = hashlib.sha256(current_encrypted_data.encode()).hexdigest()
        current_credentials = self.decrypt_api_credentials(current_encrypted_data, current_checksum)

        if not current_credentials:
            raise ValueError("Failed to decrypt current API credentials")

        # Generate new API key pair
        new_api_key, new_api_secret = self.security_manager.generate_api_key()

        # Create rotated credentials with metadata
        rotated_credentials = {
            'api_key': new_api_key,
            'api_secret': new_api_secret,
            'previous_api_key': current_credentials['api_key'],
            'rotated_at': datetime.now(timezone.utc).isoformat(),
            'rotation_reason': 'scheduled_rotation'
        }

        # Encrypt new credentials
        new_encrypted_data = self.security_manager.encrypt_sensitive_data(rotated_credentials)
        new_checksum = hashlib.sha256(new_encrypted_data.encode()).hexdigest()

        return {
            'encrypted_credentials': new_encrypted_data,
            'checksum': new_checksum
        }


# Global security manager instance
_security_manager = None
_signal_authenticator = None
_api_key_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def get_signal_authenticator() -> SignalAuthenticator:
    """Get global signal authenticator instance."""
    global _signal_authenticator
    if _signal_authenticator is None:
        _signal_authenticator = SignalAuthenticator()
    return _signal_authenticator


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def generate_hmac_signature(payload: Union[str, Dict[str, Any]]) -> str:
    """Generate HMAC signature for payload."""
    return get_security_manager().generate_hmac_signature(payload)


def verify_hmac_signature(payload: Union[str, Dict[str, Any]], signature: str) -> bool:
    """Verify HMAC signature for payload."""
    return get_security_manager().verify_hmac_signature(payload, signature)


def encrypt_sensitive_data(data: Union[str, Dict[str, Any]]) -> str:
    """Encrypt sensitive data."""
    return get_security_manager().encrypt_sensitive_data(data)


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive data."""
    return get_security_manager().decrypt_sensitive_data(encrypted_data)


def generate_jwt_token(payload: Dict[str, Any], expires_in_hours: int = 24) -> str:
    """Generate JWT token."""
    return get_security_manager().generate_jwt_token(payload, expires_in_hours)


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify JWT token."""
    return get_security_manager().verify_jwt_token(token)