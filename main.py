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

    # we are going to ignore the date column

    # let's merge together same types of features

    house_temp = ["T1_kitchen", "T2_living", "T3_laundry", "T4_office", "T5_bathroom", "T6_outside_north",
                  "T7_iron_room", "T8_teen_room", "T9_parents"]

    house_humidity = ["Hum1_kitchen", "Hum2_living", "Hum3_Laundry", "Hum4_office", "Hum5_bathroom",
                      "Hum6_outside", "Hum7_iron", "Hum8_teen_room", "Hum9_parents"]

    weather_features = ["T10_outside_WS", "Pressure_mmHg", "Hum_outside_WS", "WindSpeed_WS_m/s",
                        "Visibility_kilometer", "T_dew_point"]

    lights = ["lights_Wh"]

    random_features = ["random_1", "random_2"]

    target = ["EnergyApp_Wh"]

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    features = train[house_temp+house_humidity+weather_features+lights+random_features]

    target_feature = train[target]

    feat_description = features.describe()

    # a few observations here
    # The minimum temp inside the house is 14.89 degree celsius
    # The maximum temp inside the house is 28.29 degree celsius

    # humidity varies between 20.60% to 63.36% with exception of bathroom
    target_description = target_feature.describe()

    # according to target description 75 % of the appliances consume 100 watt or less

    features['lights_Wh'].value_counts()
    # there are 10685 zeros in total 13814 observations, this means that
    # lights are off most of the time, so we are going to ignore them
    features.drop('lights_Wh', axis=1, inplace=True)

    # lets do data visualization
    #
    # import matplotlib.pyplot as plt
    # # plot for internal temperatures
    # plt.figure()
    # for i, t in enumerate(house_temp):
    #     plt.subplot(3, 3, i+1)
    #     plt.hist(features[t], bins=25)
    #     plt.title(t)
    #
    # plt.show()
    #
    # # Observation
    # # All observations mostly follow normal distribution except
    # # T9 which is the parents rooms
    #
    # plt.figure()
    # # plot for internal humidity
    # for i, h in enumerate(house_humidity):
    #     plt.subplot(3, 3, i+1)
    #     plt.hist(features[h], bins=25)
    #     plt.title(h)
    #
    # plt.show()
    #
    # # Observation
    # # All observations mostly follow normal distribution except
    # # H6 which is humidity  outside the house
    #
    # plt.figure()
    # histogram for random features
    # for i, r in enumerate(random_features):
    #     plt.subplot(1, 2, i+1)
    #     plt.hist(features[r], bins=25)
    #     plt.title(r)
    # plt.show()
    # # The histogram for random variables show a constant freq distribution
    # # they will not effect the results
    #
    # import seaborn as sns
    # for i, feat in enumerate(weather_features):
    #     plt.subplot(2, 3, i+1)
    #     sns.distplot(features[feat], bins=10)
    #     # plt.hist(features[feat], bins=10)
    #     plt.title(feat)
    #
    # plt.show()
    #
    # # observations
    #
    # # visibility seems to be negatively skewed.
    # # outside humidity is also neg skewed
    # # wind speed is positively skewed
    #
    # plt.figure()
    # plt.subplot(1, 1, 1)
    # sns.distplot(target_feature[target[0]], bins=10)
    # plt.title(target[0])
    # plt.show()
    # # target energy appliance is positively skewed
    #
    # import numpy as np
    # # lets try to see the correlation between variables
    # train_corr = train[house_temp+house_humidity+target+random_features+weather_features]
    # corr = train_corr.corr()
    # plt.figure()
    # matrix = np.triu(corr)
    # sns.heatmap(corr, annot=True, mask=matrix, fmt=".2f")
    # # sns.heatmap(corr, annot=True, fmt="0.2f")
    # plt.xticks(range(len(corr.columns)), corr.columns);
    # # Apply yticks
    # plt.yticks(range(len(corr.columns)), corr.columns)
    # plt.show()

    # now lets drop features with low correlation

    x_train = train[features.columns]
    y_train = train[target_feature.columns]

    x_test = test[features.columns]
    y_test = test[target_feature.columns]

    to_drop = ["random_1", "random_2", "T6_outside_north"]
    x_train.drop(to_drop, axis=1, inplace=True)

    x_test.drop(to_drop, axis=1, inplace=True)

    x_train.columns

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    train = train[list(x_train.columns.values)+target]

    test = test[list(x_test.columns.values)+target]

    sc_train = pd.DataFrame(columns=train.columns, index=train.index)

    sc_train[sc_train.columns] = scaler.fit_transform(train)

    sc_test = pd.DataFrame(columns=test.columns, index=test.index)

    sc_test[sc_test.columns] = scaler.fit_transform(test)

    x_train = sc_train.drop(["EnergyApp_Wh"],axis=1)
    y_train = sc_train["EnergyApp_Wh"]

    x_test = sc_test.drop(['EnergyApp_Wh'], axis=1)
    y_test = sc_test['EnergyApp_Wh']

    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor

    from sklearn import neighbors
    from sklearn.svm import SVR

    models = [
        ['Lasso: ', Lasso()],
        ['Ridge: ', Ridge()],
        ['KNeighborsRegressor: ', neighbors.KNeighborsRegressor()],
        ['SVR:', SVR(kernel='rbf')],
        ['RandomForest ', RandomForestRegressor()],
        ['ExtraTreeRegressor :', ExtraTreesRegressor()],
        ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
        ['MLPRegressor: ', MLPRegressor(activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000,
                                        learning_rate_init=0.01, alpha=0.01)]
    ]

    import time
    from math import sqrt
    from sklearn import metrics
    from sklearn.metrics import mean_squared_error, r2_score

    model_data = []
    for name, curr_model in models:
        curr_model_data = {}
        curr_model.random_state = 78
        curr_model_data["Name"] = name
        start = time.time()
        curr_model.fit(x_train, y_train)
        end = time.time()
        curr_model_data["Train_Time"] = end - start
        curr_model_data["Train_R2_Score"] = metrics.r2_score(y_train, curr_model.predict(x_train))
        curr_model_data["Test_R2_Score"] = metrics.r2_score(y_test, curr_model.predict(x_test))
        curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(y_test, curr_model.predict(x_test)))
        model_data.append(curr_model_data)

    df = pd.DataFrame(model_data)

    df.plot(x="Name", y=['Test_R2_Score', 'Train_R2_Score', 'Test_RMSE_Score'], kind="bar", title='R2 Score Results',
            figsize=(10, 8))