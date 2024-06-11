import os
import clr

clr.AddReference(os.getcwd()+"\\I8Library1.dll")

from I8Devices import Device,Settings
device = Device()
device_settings = Settings()