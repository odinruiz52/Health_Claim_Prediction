"""
🏥 Health Insurance Claim Prediction - Configuration Management

This module provides centralized configuration management for the entire
health insurance claim prediction system.

Features:
- Environment-based configuration
- Validation of configuration values
- Default values for all settings
- Secure handling of sensitive information
- Healthcare-specific configurations
"""

import os
import logging
import secrets
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    model_path: str = "models/final_pipeline_classification.joblib"
    scaler_path: str = "models/scaler.pkl"  # Legacy - no longer used with Pipeline
    model_version: str = "2.0.0"
    prediction_threshold: float = 0.5
    
    # Feature validation ranges
    age_min: int = 18
    age_max: int = 100
    bmi_min: float = 15.0
    bmi_max: float = 50.0
    children_min: int = 0
    children_max: int = 10
    charges_min: float = 0.0
    charges_max: float = 100000.0


@dataclass
class DataConfig:
    """Data-related configuration settings."""
    data_path: str = "data/insurance_encoded.csv"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Data quality thresholds
    missing_threshold: float = 0.05  # 5% missing allowed
    outlier_threshold: float = 0.1   # 10% outliers allowed


@dataclass
class FlaskConfig:
    """Flask web application configuration."""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 5000
    secret_key: str = "dev-key-change-in-production"
    
    # Security settings
    max_content_length: int = 16 * 1024 * 1024  # 16MB
    session_timeout: int = 3600  # 1 hour


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class ValidationConfig:
    """Input validation configuration constants."""
    # These constants are used by both Flask app and CLI for consistency
    AGE_MIN: int = 18
    AGE_MAX: int = 100
    BMI_MIN: float = 15.0
    BMI_MAX: float = 50.0
    CHILDREN_MIN: int = 0
    CHILDREN_MAX: int = 10
    CHARGES_MIN: float = 0.0
    CHARGES_MAX: float = 100000.0
    
    # Valid categorical values
    VALID_SEX_VALUES: tuple = (0, 1)
    VALID_SMOKER_VALUES: tuple = (0, 1)
    VALID_REGION_VALUES: tuple = (0, 1, 2, 3)


@dataclass
class HealthcareConfig:
    """Healthcare-specific configuration."""
    # Compliance settings
    hipaa_logging: bool = True
    audit_trail: bool = True
    data_retention_days: int = 2555  # 7 years
    
    # Business rules
    high_risk_age: int = 60
    obesity_bmi_threshold: float = 30.0
    high_cost_percentile: float = 0.75
    
    # Ethical AI settings
    fairness_check: bool = True
    bias_monitoring: bool = True
    explainability_required: bool = True


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, environment: str = None):
        """Initialize configuration based on environment."""
        self.environment = environment or os.getenv("FLASK_ENV", "development")
        
        # Initialize configuration sections
        self.model = ModelConfig()
        self.data = DataConfig()
        self.flask = FlaskConfig()
        self.logging = LoggingConfig()
        self.healthcare = HealthcareConfig()
        self.validation = ValidationConfig()  # Add validation config
        
        # Load environment-specific configurations
        self._load_environment_config()
        self._validate_config()
        self._setup_directories()
    
    def _load_environment_config(self) -> None:
        """Load configuration from environment variables."""
        
        # Model configuration from environment
        self.model.model_path = os.getenv("MODEL_PATH", self.model.model_path)
        self.model.scaler_path = os.getenv("SCALER_PATH", self.model.scaler_path)
        self.model.prediction_threshold = float(
            os.getenv("PREDICTION_THRESHOLD", self.model.prediction_threshold)
        )
        
        # Data configuration from environment
        self.data.data_path = os.getenv("DATA_PATH", self.data.data_path)
        self.data.test_size = float(os.getenv("TEST_SIZE", self.data.test_size))
        
        # Flask configuration from environment
        self.flask.debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
        self.flask.host = os.getenv("FLASK_HOST", self.flask.host)
        self.flask.port = int(os.getenv("FLASK_PORT", self.flask.port))
        self.flask.secret_key = os.getenv("SECRET_KEY", self._generate_secret_key())
        
        # Logging configuration from environment
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file_path = os.getenv("LOG_FILE", self.logging.file_path)
        
        # Healthcare configuration from environment
        self.healthcare.hipaa_logging = (
            os.getenv("HIPAA_LOGGING", "True").lower() == "true"
        )
        self.healthcare.audit_trail = (
            os.getenv("AUDIT_TRAIL", "True").lower() == "true"
        )
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for Flask sessions."""
        if self.environment == "production":
            # In production, this should come from environment variable
            secret_key = os.getenv("SECRET_KEY")
            if not secret_key:
                raise ValueError("SECRET_KEY environment variable must be set in production")
            return secret_key
        else:
            # Generate a secure development key using secrets module
            return secrets.token_hex(32)
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        
        # Validate model configuration
        if not (0 < self.model.prediction_threshold < 1):
            raise ValueError("Prediction threshold must be between 0 and 1")
        
        if self.model.age_min >= self.model.age_max:
            raise ValueError("Age minimum must be less than age maximum")
        
        if self.model.bmi_min >= self.model.bmi_max:
            raise ValueError("BMI minimum must be less than BMI maximum")
        
        # Validate data configuration
        if not (0 < self.data.test_size < 1):
            raise ValueError("Test size must be between 0 and 1")
        
        if self.data.cv_folds < 2:
            raise ValueError("Cross-validation folds must be at least 2")
        
        # Validate Flask configuration
        if not (1024 <= self.flask.port <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        
        # Validate logging configuration
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.level not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            "models",
            "data", 
            "logs",
            "results",
            "plots"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get database URL (for future database integration)."""
        return os.getenv("DATABASE_URL", "sqlite:///health_insurance.db")
    
    def get_redis_url(self) -> str:
        """Get Redis URL (for future caching integration)."""
        return os.getenv("REDIS_URL", "redis://localhost:6379")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "model": {
                "model_path": self.model.model_path,
                "scaler_path": self.model.scaler_path,
                "model_version": self.model.model_version,
                "prediction_threshold": self.model.prediction_threshold
            },
            "data": {
                "data_path": self.data.data_path,
                "test_size": self.data.test_size,
                "random_state": self.data.random_state,
                "cv_folds": self.data.cv_folds
            },
            "flask": {
                "debug": self.flask.debug,
                "host": self.flask.host,
                "port": self.flask.port
            },
            "healthcare": {
                "hipaa_logging": self.healthcare.hipaa_logging,
                "audit_trail": self.healthcare.audit_trail,
                "fairness_check": self.healthcare.fairness_check
            }
        }
    
    def print_config(self) -> None:
        """Print current configuration (excluding sensitive data)."""
        config_dict = self.to_dict()
        
        print("🏥 Health Insurance Claim Prediction - Configuration")
        print("=" * 60)
        
        for section, values in config_dict.items():
            print(f"\\n[{section.upper()}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    # Hide sensitive information
                    if "key" in key.lower() or "secret" in key.lower():
                        value = "*" * len(str(value)) if value else "Not set"
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")
        
        print("\\n" + "=" * 60)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def setup_logging(config: Config) -> None:
    """Setup logging based on configuration."""
    
    # Create logs directory
    log_dir = Path(config.logging.file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file_path),
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    def __init__(self):
        super().__init__("development")
        self.flask.debug = True
        self.logging.level = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""
    
    def __init__(self):
        super().__init__("production")
        self.flask.debug = False
        self.logging.level = "INFO"
        self.healthcare.audit_trail = True
        self.healthcare.hipaa_logging = True


class TestingConfig(Config):
    """Testing environment configuration."""
    
    def __init__(self):
        super().__init__("testing")
        self.flask.debug = False
        self.logging.level = "WARNING"
        self.data.data_path = "tests/test_data.csv"


# Configuration factory
def create_config(environment: str = None) -> Config:
    """Create configuration instance based on environment."""
    
    if environment == "production":
        return ProductionConfig()
    elif environment == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


if __name__ == "__main__":
    # Print current configuration
    config = get_config()
    config.print_config()