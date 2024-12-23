import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (SimpleImputer, IterativeImputer)
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from sklearn.model_selection import (GridSearchCV, cross_val_score)
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

# 加载数据
full_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 分析数据
print(full_df.head())
print(full_df.info())
# 将 'Transported' 列转换为数值类型
full_df['Transported'] = full_df['Transported'].astype(int)

# 图形显示，确保只使用数值列计算相关性
numeric_cols = full_df.select_dtypes(include=[np.number]).columns
print(full_df[numeric_cols].corr()['Transported'].sort_values(ascending=False))

full_df.hist(bins=30, figsize=(12, 8))
plt.show()

# 分离特征和目标
y = full_df.Transported.astype(int)
df = full_df.drop(['Transported'], axis=1).copy()
df = pd.concat([df, test_df], axis=0).reset_index(drop=True)
test_pass_id = test_df.PassengerId.copy()
X_max_index = full_df.shape[0]

# 处理异常值
df['Pass_group'] = df.PassengerId.str.split('_').str[0].astype(float)
df['Lastname'] = df.Name.str.split(' ').str[1]
df[['Deck', 'Cab_num', 'Deck_side']] = df.Cabin.str.split('/', expand=True)
df.Cab_num = df.Cab_num.astype(float)

# 限制异常值
df.loc[df.RoomService.gt(9000), 'RoomService'] = 9000
df.loc[df.FoodCourt.gt(22000), 'FoodCourt'] = 22000
df.loc[df.ShoppingMall.gt(11000), 'ShoppingMall'] = 11000
df.loc[df.Spa.gt(17000), 'Spa'] = 17000
df.loc[df.VRDeck.gt(21000), 'VRDeck'] = 21000

# 补全设施列
amenities = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df.loc[df.CryoSleep.eq(True), amenities] = 0
zero_amenities = df[amenities].sum(axis=1).eq(0)
df.loc[zero_amenities, amenities] = 0
for i in amenities:
    df.loc[df[i].isna(), i] = df.loc[df[i].gt(0), i].median()

# 创建总费用列
df['Total_expenses'] = df[amenities].sum(axis=1)
df.loc[(df.CryoSleep.isna() & df.Total_expenses.gt(0)), 'CryoSleep'] = False

# 补全VIP列
df.loc[(df.VIP.isna() & (df.Age < 18)), 'VIP'] = False
df.loc[(df.VIP.isna() & (df.HomePlanet == 'Earth')), 'VIP'] = False
df.loc[(df.VIP.isna() & (df.HomePlanet.eq('Mars')) & (df.Destination.eq('55 Cancri e'))), 'VIP'] = False
df.loc[(df.VIP.isna() & df.Deck.isin(['G', 'T'])), 'VIP'] = False
df.loc[df.VIP.isna() & df.CryoSleep.eq(False) & ~df.Deck.isin(['A', 'B', 'C', 'D']), 'VIP'] = True

# 补全HomePlanet列
df.loc[(df.HomePlanet.isna() & df.VIP.eq(True) & df.Destination.eq('55 Cancri e')), 'HomePlanet'] = 'Europa'
group_home_map = (df.loc[~df.Pass_group.isna() & ~df.HomePlanet.isna(), ['Pass_group', 'HomePlanet']]
                  .set_index('Pass_group').to_dict()['HomePlanet'])
df.loc[df.HomePlanet.isna(), 'HomePlanet'] = df.Pass_group.map(group_home_map)
df.loc[(df.HomePlanet.isna() & df.Deck.isin(['T', 'A', 'B', 'C'])), 'HomePlanet'] = 'Europa'
df.loc[(df.HomePlanet.isna() & df.Deck.eq('G')), 'HomePlanet'] = 'Earth'
lastname_home_map = (df.loc[~df.Lastname.isna() & ~df.HomePlanet.isna(), ['Lastname', 'HomePlanet']]
                     .set_index('Lastname').to_dict()['HomePlanet'])
df.loc[df.HomePlanet.isna(), 'HomePlanet'] = df.Lastname.map(lastname_home_map)

# 补全Age列
df.loc[((df.VIP == True) & df.Age.isna()), 'Age'] = df.loc[(df.VIP == True), 'Age'].median()
df.loc[(df.Age.isna() & df.Total_expenses.gt(0)), 'Age'] = df.loc[df.Total_expenses.gt(0), 'Age'].median()
df.loc[(df.Age.isna() & df.Total_expenses.eq(0) & df.CryoSleep.eq(False)), 'Age'] = df.loc[(df.Total_expenses.eq(0) & df.CryoSleep.eq(False)), 'Age'].median()
df.Age.fillna(df.Age.median(), inplace=True)

# 补全Cab_num列
df.Cab_num.fillna(df.Cab_num.median(), inplace=True)

# 创建Group_members和Cabin_members列
Group_members = df.Pass_group.value_counts().to_dict()
df['Group_members'] = df.Pass_group.map(Group_members)
Cabin_members = df.Cabin.value_counts().to_dict()
df['Cabin_members'] = df.Cabin.map(Cabin_members)
df.Cabin_members.fillna(df.Cabin_members.mean(), inplace=True)

# 创建Deck_transp_ratio和Deck_side_transp_ratio列
X = df[:X_max_index]
test_df = df[X_max_index:]
full_df = pd.concat([X, y], axis=1).copy()
deck_total_pass = full_df.groupby('Deck').Deck.count()
deck_total_transported = full_df.groupby('Deck').Transported.sum()
Deck_transp_ratio = (deck_total_transported / deck_total_pass).to_dict()
df['Deck_transp_ratio'] = df.Deck.map(Deck_transp_ratio)
df.Deck_transp_ratio.fillna(df.Deck_transp_ratio.mean(), inplace=True)
deck_side_total = full_df.groupby('Deck_side').Deck.count()
deck_side_transported = full_df.groupby('Deck_side').Transported.sum()
Deck_side_transp_ratio = (deck_side_transported / deck_side_total).to_dict()
df['Deck_side_transp_ratio'] = df.Deck_side.map(Deck_side_transp_ratio)
df.Deck_side_transp_ratio.fillna(df.Deck_side_transp_ratio.mean(), inplace=True)

# 删除不用的列
col_drop = ['PassengerId', 'Cabin', 'Name', 'Lastname']
df = df.drop(col_drop, axis=1)

# 补全和编码类别特征
categ_cols = list(df.select_dtypes(['object', 'category']).columns)
cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
df_cat = pd.DataFrame(cat_imputer.fit_transform(df[categ_cols]), columns=df[categ_cols].columns)
df_cat = pd.get_dummies(df_cat)

# 补全数值特征
num_cols = list(df.select_dtypes(['int64', 'float64']).columns)
df = pd.concat([df_cat, df[num_cols]], axis=1)
it_imp = IterativeImputer()
df = pd.DataFrame(it_imp.fit_transform(df), columns=df.columns)

df['Age_group'] = pd.cut(x=df.Age, labels=[1, 3, 2], bins=[-1, 17, 43, df.Age.max()]).astype('float')
df['Total_expenses_group'] = pd.cut(x=df.Total_expenses, labels=[3, 1, 2], bins=[-1, 1, 2250, df.Total_expenses.max()]).astype('float')
df['Cab_group'] = pd.cut(x=df.Cab_num, labels=[3, 2, 4, 1], bins=[-1, 300, 690, 1170, df.Cab_num.max()]).astype('float')
df['Pass_group_type'] = pd.cut(x=df.Pass_group, labels=[2, 3, 1], bins=[-1, 3400, 7300, df.Pass_group.max()]).astype('float')

# 标准化
skewed_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_expenses']
df[skewed_features] = df[skewed_features].apply(np.log1p)
std_scaler = StandardScaler()
df_scaled = std_scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)

# 删除不用于建模的列
col_drop = ['Cab_num', 'Pass_group']
df = df.drop(col_drop, axis=1)

# 分离训练和测试数据
X = df[:X_max_index]
test_df = df[X_max_index:]

# 找到最佳特征
final_features = [
    'HomePlanet_Earth', 'HomePlanet_Mars', 'HomePlanet_Missing', 'CryoSleep_True',
    'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e', 'Deck_A', 'Deck_Missing',
    'Deck_T', 'Deck_side_P', 'Age', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck',
    'Total_expenses', 'Group_members', 'Deck_transp_ratio', 'Total_expenses_group', 'Cab_group']

# CatBoost网格搜索参数调优
params =  {'depth': 6,
           'iterations': 2000,
           'learning_rate': 0.01,
           'thread_count': -1,
           'verbose': False}

# 最终模型
cat_model = CatBoostClassifier(**params)
cat_model.fit(X[final_features], y)
cat_rmses = cross_val_score(cat_model, X[final_features], y, cv=5)
print(pd.Series(cat_rmses).describe())
print('\n', cat_model.get_feature_importance(prettified=True))

# 提交
test_preds = cat_model.predict(test_df[final_features])
output = pd.DataFrame({'PassengerId': test_pass_id, 'Transported': test_preds.astype(bool)})
output.to_csv('submission_r.csv', index=False)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
#
# # 1. 查看数据
# train_data = pd.read_csv('../train.csv')
# train_data.head()
#
# # 2. 特征处理
# # 拆分舱位
# train_data[['deck', 'num', 'side']] = train_data['Cabin'].str.split('/', expand=True)
#
# # 填充空值
# train_data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train_data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
#
# # 类型转换
# train_data['Transported'] = train_data['Transported'].astype(int)
# train_data['VIP'] = train_data['VIP'].astype(int)
# train_data['CryoSleep'] = train_data['CryoSleep'].astype(int)
# train_data[['HomePlanet', 'Destination', 'deck', 'side']] = train_data[['HomePlanet', 'Destination', 'deck', 'side']].astype('category')
# train_data['num'] = pd.to_numeric(train_data['num'], errors='coerce').astype('Int64')
#
# # 独热编码
# category_data = pd.get_dummies(train_data[['HomePlanet', 'Destination', 'deck', 'side']])
#
# # 删除不需要的特征
# train_data = pd.concat([train_data.drop(['HomePlanet', 'Destination', 'deck', 'side'], axis=1), category_data], axis=1)
# train_data = train_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
#
# # 3. 训练数据和标签
# y = train_data['Transported']
# X = train_data.drop(['Transported'], axis=1)
#
# # 4. 拆分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=20)
#
# # 5. 网格搜索进行超参数调优
# xgb_model = XGBClassifier()
#
# param_grid = {
#     "max_depth": [3, 4, 5, 6, 7],
#     "learning_rate": [0.1, 0.07, 0.05, 0.03, 0.01],
#     'n_estimators': [10, 50, 100, 200, 300]
# }
#
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
#     scoring='accuracy'
# )
#
# grid_search.fit(X_train, y_train)
#
# # 输出最佳参数和得分
# print("Best parameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
#
# # 6. 测试集预测与评估
# y_pred = grid_search.best_estimator_.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# # 7. 验证超参（可选）
# xgb_model = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=200, eval_metric=['error'])
# xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=True)
#
# y_pred = xgb_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# # 8. 查看特征的重要性
# feature_importance_map = zip(X_train.columns, grid_search.best_estimator_.feature_importances_)
# sorted_feature_importance = sorted(feature_importance_map, key=lambda x: x[1], reverse=True)
#
# # 打印排序后的特征和其重要性
# for feature, importance in sorted_feature_importance:
#     print(f"Feature {feature}: Importance = {importance}")
#
# # 9. 提交预测结果
# test_data = pd.read_csv('../test.csv')
# passenger_ids = test_data.PassengerId
#
# test_data[['deck', 'num', 'side']] = test_data['Cabin'].str.split('/', expand=True)
# test_data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = test_data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
# test_data['VIP'] = test_data['VIP'].astype(int)
# test_data['CryoSleep'] = test_data['CryoSleep'].astype(int)
# test_data[['HomePlanet', 'Destination', 'deck', 'side']] = test_data[['HomePlanet', 'Destination', 'deck', 'side']].astype('category')
# test_data['num'] = pd.to_numeric(test_data['num'], errors='coerce').astype('Int64')
#
# # 独热编码
# category_data = pd.get_dummies(test_data[['HomePlanet', 'Destination', 'deck', 'side']])
# test_data = pd.concat([test_data.drop(['HomePlanet', 'Destination', 'deck', 'side'], axis=1), category_data], axis=1)
#
# # 删除不需要的特征
# test_data = test_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
#
# # 预测结果
# y_pred = grid_search.best_estimator_.predict(test_data).astype(bool)
#
# # 保存结果
# output_df = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': y_pred})
# output_df.to_csv('submission.csv', index=False)
#
