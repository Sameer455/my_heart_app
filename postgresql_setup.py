# PostgreSQL Setup Script for Heart Disease App
# Run this script to set up PostgreSQL database

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_postgresql():
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host="localhost",
            user="postgres",
            password="password"  # Change this to your PostgreSQL password
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE heart_disease_db;")
        print("✅ Database 'heart_disease_db' created successfully!")
        
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL setup completed!")
        print("📝 Update your Django settings.py with:")
        print("""
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'heart_disease_db',
        'USER': 'postgres',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
        """)
        
    except Exception as e:
        print(f"❌ Error setting up PostgreSQL: {e}")
        print("💡 Make sure PostgreSQL is installed and running")

if __name__ == "__main__":
    setup_postgresql()
