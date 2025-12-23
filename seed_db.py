import random
import sqlite3

from faker import Faker

from src.engine.db import DBManager
from src.schema import AETHERIS_DB


def seed_database(rows_per_table: int = 100):
    db = DBManager(AETHERIS_DB)
    db.setup_db()
    faker = Faker()

    rooms = ["Kitchen", "Living Room", "Garage", "Bedroom", "Home Cinema"]
    devices = ["Lamp", "Sensor", "Thermostat", "Camera"]

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        for _ in range(rows_per_table):
            cursor.execute(
                'INSERT INTO "climate_stats" (room, timestamp, temp, humidity, co2) VALUES (?, ?, ?, ?, ?)',
                (
                    random.choice(rooms),
                    faker.date_time_this_year().isoformat(),
                    random.uniform(18, 30),
                    random.uniform(30, 60),
                    random.randint(400, 1000),
                ),
            )
            cursor.execute(
                'INSERT INTO "energy_consumption" (device_id, timestamp, kwh, voltage) VALUES (?, ?, ?, ?)',
                (
                    f"{random.choice(devices)}_{random.randint(1, 5)}",
                    faker.date_time_this_year().isoformat(),
                    random.uniform(0.1, 5.0),
                    220.0,
                ),
            )
        conn.commit()
    print(f"Database seeded with {rows_per_table} rows per table.")


if __name__ == "__main__":
    seed_database()
