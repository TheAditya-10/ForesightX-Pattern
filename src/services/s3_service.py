"""
ForesightX S3 Service Module
=============================
Production-grade AWS S3 connection and operations module.

Features:
- Environment variable configuration
- Secure credential management
- Upload/Download operations
- Bucket management
- Error handling and retry logic
- Progress tracking
- Logging integration
"""

import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Dict, Any

# Handle both direct execution and module import
try:
    from .logger import get_logger
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.services.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)


class S3Service:
    """
    AWS S3 Service for file operations
    
    Environment Variables Required:
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
        AWS_REGION: AWS region (default: us-east-1)
        S3_BUCKET_NAME: Default S3 bucket name
    """
    
    def __init__(self, 
                 bucket_name: Optional[str] = None,
                 region: Optional[str] = None):
        """
        Initialize S3 service with credentials from environment
        
        Args:
            bucket_name: S3 bucket name (overrides env variable)
            region: AWS region (overrides env variable)
        """
        # Load credentials from environment
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        
        # Validate credentials
        if not self.access_key or not self.secret_key:
            logger.error("AWS credentials not found in environment variables")
            raise ValueError(
                "AWS credentials not configured. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
            )
        
        if not self.bucket_name:
            logger.warning("S3_BUCKET_NAME not set. You'll need to specify bucket for operations.")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            logger.info(f"S3 client initialized successfully for region: {self.region}")
            
            # Initialize S3 resource for advanced operations
            self.s3_resource = boto3.resource(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test S3 connection by listing buckets
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing S3 connection...")
            response = self.s3_client.list_buckets()
            bucket_count = len(response['Buckets'])
            logger.info(f"✓ S3 connection successful! Found {bucket_count} buckets")
            return True
        except NoCredentialsError:
            logger.error("AWS credentials not found or invalid")
            return False
        except ClientError as e:
            logger.error(f"Failed to connect to S3: {e.response['Error']['Message']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing S3 connection: {str(e)}")
            return False
    
    def list_buckets(self) -> List[str]:
        """
        List all S3 buckets
        
        Returns:
            List of bucket names
        """
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            logger.info(f"Found {len(buckets)} S3 buckets")
            return buckets
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e.response['Error']['Message']}")
            return []
    
    def bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """
        Check if bucket exists
        
        Args:
            bucket_name: Bucket name (uses default if not provided)
            
        Returns:
            True if bucket exists, False otherwise
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided")
            return False
        
        try:
            self.s3_client.head_bucket(Bucket=bucket)
            logger.debug(f"Bucket '{bucket}' exists")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.debug(f"Bucket '{bucket}' does not exist")
            else:
                logger.error(f"Error checking bucket '{bucket}': {e.response['Error']['Message']}")
            return False
    
    def create_bucket(self, bucket_name: str) -> bool:
        """
        Create a new S3 bucket
        
        Args:
            bucket_name: Name for the new bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"✓ Bucket '{bucket_name}' created successfully")
            return True
        except ClientError as e:
            logger.error(f"Failed to create bucket '{bucket_name}': {e.response['Error']['Message']}")
            return False
    
    def upload_file(self, 
                    local_path: str, 
                    s3_key: str,
                    bucket_name: Optional[str] = None,
                    metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload a file to S3
        
        Args:
            local_path: Path to local file
            s3_key: S3 object key (path in bucket)
            bucket_name: Bucket name (uses default if not provided)
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided for upload")
            return False
        
        local_file = Path(local_path)
        if not local_file.exists():
            logger.error(f"Local file not found: {local_path}")
            return False
        
        try:
            logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
            
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.upload_file(
                str(local_file),
                bucket,
                s3_key,
                ExtraArgs=extra_args if extra_args else None
            )
            
            logger.info(f"✓ Upload successful: s3://{bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload file: {e.response['Error']['Message']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during upload: {str(e)}")
            return False
    
    def download_file(self,
                     s3_key: str,
                     local_path: str,
                     bucket_name: Optional[str] = None) -> bool:
        """
        Download a file from S3
        
        Args:
            s3_key: S3 object key
            local_path: Local path to save file
            bucket_name: Bucket name (uses default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided for download")
            return False
        
        try:
            # Create directory if it doesn't exist
            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path}")
            
            self.s3_client.download_file(bucket, s3_key, str(local_file))
            
            logger.info(f"✓ Download successful: {local_path}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"File not found in S3: s3://{bucket}/{s3_key}")
            else:
                logger.error(f"Failed to download file: {e.response['Error']['Message']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {str(e)}")
            return False
    
    def list_objects(self,
                    prefix: str = '',
                    bucket_name: Optional[str] = None) -> List[str]:
        """
        List objects in S3 bucket
        
        Args:
            prefix: Prefix filter for objects
            bucket_name: Bucket name (uses default if not provided)
            
        Returns:
            List of object keys
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided")
            return []
        
        try:
            logger.info(f"Listing objects in s3://{bucket}/{prefix}")
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.info("No objects found")
                return []
            
            objects = [obj['Key'] for obj in response['Contents']]
            logger.info(f"Found {len(objects)} objects")
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list objects: {e.response['Error']['Message']}")
            return []
    
    def delete_object(self,
                     s3_key: str,
                     bucket_name: Optional[str] = None) -> bool:
        """
        Delete an object from S3
        
        Args:
            s3_key: S3 object key
            bucket_name: Bucket name (uses default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided")
            return False
        
        try:
            logger.info(f"Deleting s3://{bucket}/{s3_key}")
            
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            
            logger.info(f"✓ Object deleted: s3://{bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete object: {e.response['Error']['Message']}")
            return False
    
    def get_object_metadata(self,
                           s3_key: str,
                           bucket_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an S3 object
        
        Args:
            s3_key: S3 object key
            bucket_name: Bucket name (uses default if not provided)
            
        Returns:
            Metadata dictionary or None if failed
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided")
            return None
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            
            metadata = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'metadata': response.get('Metadata', {})
            }
            
            logger.debug(f"Retrieved metadata for s3://{bucket}/{s3_key}")
            return metadata
            
        except ClientError as e:
            logger.error(f"Failed to get metadata: {e.response['Error']['Message']}")
            return None
    
    def upload_dataframe(self,
                        df,
                        s3_key: str,
                        bucket_name: Optional[str] = None,
                        file_format: str = 'csv') -> bool:
        """
        Upload pandas DataFrame to S3
        
        Args:
            df: Pandas DataFrame
            s3_key: S3 object key
            bucket_name: Bucket name (uses default if not provided)
            file_format: File format ('csv' or 'parquet')
            
        Returns:
            True if successful, False otherwise
        """
        import tempfile
        
        bucket = bucket_name or self.bucket_name
        if not bucket:
            logger.error("No bucket name provided")
            return False
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{file_format}', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
                if file_format == 'csv':
                    df.to_csv(tmp_path, index=False)
                elif file_format == 'parquet':
                    df.to_parquet(tmp_path, index=False)
                else:
                    logger.error(f"Unsupported format: {file_format}")
                    return False
            
            success = self.upload_file(tmp_path, s3_key, bucket)
            
            # Cleanup temp file
            os.unlink(tmp_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("Testing S3 Service")
    print("="*80)
    
    try:
        # Initialize S3 service
        s3 = S3Service()
        
        # Test connection
        if s3.test_connection():
            print("\n✓ S3 connection successful!")
            
            # List buckets
            buckets = s3.list_buckets()
            print(f"\n📦 Available buckets: {buckets}")
            
        else:
            print("\n✗ S3 connection failed!")
            print("\nPlease ensure:")
            print("1. AWS credentials are set in .env file:")
            print("   AWS_ACCESS_KEY_ID=your_access_key")
            print("   AWS_SECRET_ACCESS_KEY=your_secret_key")
            print("   AWS_REGION=us-east-1")
            print("   S3_BUCKET_NAME=your_bucket_name")
            
    except ValueError as e:
        print(f"\n✗ Configuration Error: {str(e)}")
        print("\nCreate a .env file with your AWS credentials:")
        print("AWS_ACCESS_KEY_ID=your_access_key")
        print("AWS_SECRET_ACCESS_KEY=your_secret_key")
        print("AWS_REGION=us-east-1")
        print("S3_BUCKET_NAME=your_bucket_name")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    
    print("\n" + "="*80)
