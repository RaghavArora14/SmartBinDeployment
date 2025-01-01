import sqlite3

def initialize_db():
    # Connect to the database (it will create one if it doesn't exist)
    conn = sqlite3.connect('waste_data.db')
    cursor = conn.cursor()
    
    # Create table for classifications
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            is_biodegradable BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    initialize_db()
