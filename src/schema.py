from enum import Enum

from pydantic import BaseModel


class ColumnType(str, Enum):
    TEXT = "TEXT"
    REAL = "REAL"
    INTEGER = "INTEGER"
    DATETIME = "DATETIME"
    DATE = "DATE"


class ColumnSchema(BaseModel):
    name: str
    type: ColumnType
    description: str


class TableSchema(BaseModel):
    name: str
    description: str
    columns: list[ColumnSchema]

    @property
    def column_names(self) -> list[str]:
        """Returns a list of column names."""
        return [col.name for col in self.columns]


class AetherisSchema(BaseModel):
    tables: list[TableSchema]

    def get_table(self, name: str) -> TableSchema | None:
        """Finds a table by name."""
        return next((t for t in self.tables if t.name == name), None)


AETHERIS_DB = AetherisSchema(
    tables=[
        TableSchema(
            name="devices",
            description="Registry of all smart devices in the house.",
            columns=[
                ColumnSchema(name="id", type=ColumnType.TEXT, description="Unique identifier for the hardware device"),
                ColumnSchema(name="name", type=ColumnType.TEXT, description="Human-readable name assigned to the device"),
                ColumnSchema(name="room", type=ColumnType.TEXT, description="Location/Room where the device is installed"),
                ColumnSchema(name="type", type=ColumnType.TEXT, description="Category of device (e.g., sensor, actuator, light)"),
                ColumnSchema(name="manufacturer", type=ColumnType.TEXT, description="Company that produced the device"),
                ColumnSchema(name="status", type=ColumnType.TEXT, description="Current operational state (online, offline, standby)"),
            ],
        ),
        TableSchema(
            name="energy_consumption",
            description="Electricity usage metrics for devices.",
            columns=[
                ColumnSchema(name="device_id", type=ColumnType.TEXT, description="Reference to the device being monitored"),
                ColumnSchema(name="timestamp", type=ColumnType.DATETIME, description="Exact time of the energy reading"),
                ColumnSchema(name="kwh", type=ColumnType.REAL, description="Energy consumed since last reading in kilowatt-hours"),
                ColumnSchema(name="voltage", type=ColumnType.REAL, description="Main line voltage measured at the device plug"),
            ],
        ),
        TableSchema(
            name="climate_stats",
            description="Indoor climate and air quality metrics.",
            columns=[
                ColumnSchema(name="room", type=ColumnType.TEXT, description="Room where the climate sensor is located"),
                ColumnSchema(name="timestamp", type=ColumnType.DATETIME, description="Time of environmental data capture"),
                ColumnSchema(name="temp", type=ColumnType.REAL, description="Ambient temperature in degrees Celsius"),
                ColumnSchema(name="humidity", type=ColumnType.REAL, description="Relative humidity percentage (0-100%)"),
                ColumnSchema(name="co2", type=ColumnType.INTEGER, description="Carbon dioxide concentration in ppm (parts per million)"),
            ],
        ),
        TableSchema(
            name="security_logs",
            description="Security events and access logs.",
            columns=[
                ColumnSchema(name="timestamp", type=ColumnType.DATETIME, description="Event occurrence time"),
                ColumnSchema(name="event_type", type=ColumnType.TEXT, description="Type of event (e.g., motion detected, door opened)"),
                ColumnSchema(name="severity", type=ColumnType.TEXT, description="Alert level: low, medium, high, or critical"),
                ColumnSchema(name="authorized_person", type=ColumnType.TEXT, description="Name of the person identified, if applicable"),
            ],
        ),
        TableSchema(
            name="lighting_states",
            description="States of smart lighting including brightness and color.",
            columns=[
                ColumnSchema(name="device_id", type=ColumnType.TEXT, description="ID of the smart bulb or light strip"),
                ColumnSchema(name="brightness", type=ColumnType.INTEGER, description="Current light intensity (0-100)"),
                ColumnSchema(name="color_temp", type=ColumnType.INTEGER, description="Color temperature in Kelvin"),
                ColumnSchema(name="mode", type=ColumnType.TEXT, description="Active lighting scene (e.g., movie, reading, night)"),
            ],
        ),
        TableSchema(
            name="water_usage",
            description="Water consumption and leak detection.",
            columns=[
                ColumnSchema(name="sensor_id", type=ColumnType.TEXT, description="ID of the water flow meter"),
                ColumnSchema(name="timestamp", type=ColumnType.DATETIME, description="Time of measurement"),
                ColumnSchema(name="liters", type=ColumnType.REAL, description="Total volume of water used in the interval"),
                ColumnSchema(name="flow_rate", type=ColumnType.REAL, description="Current speed of water flow in liters per minute"),
            ],
        ),
        TableSchema(
            name="occupancy",
            description="Room occupancy and activity levels.",
            columns=[
                ColumnSchema(name="room", type=ColumnType.TEXT, description="Room being monitored"),
                ColumnSchema(name="timestamp", type=ColumnType.DATETIME, description="Measurement time"),
                ColumnSchema(name="person_count", type=ColumnType.INTEGER, description="Estimated number of people present"),
                ColumnSchema(
                    name="activity_level",
                    type=ColumnType.TEXT,
                    description="Level of movement detected (stationary, active, high)",
                ),
            ],
        ),
        TableSchema(
            name="appliance_health",
            description="Maintenance and health status of home appliances.",
            columns=[
                ColumnSchema(name="device_id", type=ColumnType.TEXT, description="ID of the appliance (e.g., vacuum, air purifier)"),
                ColumnSchema(name="battery_level", type=ColumnType.INTEGER, description="Remaining battery charge percentage"),
                ColumnSchema(name="filter_life", type=ColumnType.INTEGER, description="Remaining consumable life percentage"),
                ColumnSchema(name="last_service", type=ColumnType.DATE, description="Date of the last professional maintenance"),
            ],
        ),
        TableSchema(
            name="network_traffic",
            description="Internet and local network usage per device.",
            columns=[
                ColumnSchema(name="device_id", type=ColumnType.TEXT, description="ID of the connected device"),
                ColumnSchema(name="bandwidth_up", type=ColumnType.REAL, description="Current upload speed in Mbps"),
                ColumnSchema(name="bandwidth_down", type=ColumnType.REAL, description="Current download speed in Mbps"),
                ColumnSchema(name="signal_strength", type=ColumnType.INTEGER, description="Wi-Fi signal strength in dBm or percentage"),
            ],
        ),
        TableSchema(
            name="voice_commands",
            description="History of voice assistant interactions.",
            columns=[
                ColumnSchema(name="timestamp", type=ColumnType.DATETIME, description="When the command was issued"),
                ColumnSchema(name="room", type=ColumnType.TEXT, description="Room where the voice was captured"),
                ColumnSchema(name="command_text", type=ColumnType.TEXT, description="Transcribed text of the voice command"),
                ColumnSchema(name="success_rate", type=ColumnType.REAL, description="Confidence level or execution success score"),
            ],
        ),
    ],
)
