import pandas as pd
if __name__ == "__main__":
    header = ["datetime", "EnergyApp_Wh", "lights_Wh", "T1_kitchen", "Hum1_kitchen", "T2_living", "Hum2_living",
              "T3_laundry", "Hum3_Laundry", "T4_office", "Hum4_office", "T5_bathroom", "Hum5_bathroom",
              "T6_outside_north", "Hum6_outside", "T7_iron_room", "Hum7_iron", "T8_teen_room", "Hum8_teen_room",
              "T9_parents", "Hum9_parents", "T10_outside_WS", "Pressure_mmHg", "Hum_outside_WS", "WindSpeed_WS_m/s",
              "Visibility_kilometer", "T_dew_point", "random_1", "random_2"]
    len(header)

    df = pd.read_csv("data.csv")


    df.columns = header

    print(df.head())
    df.info()

    df.describe()



