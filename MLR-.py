{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6c95d23b-84b7-4f58-8e3c-d64e7bf328fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler,LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression,Lasso,Ridge\n",
    "from sklearn.metrics import r2_score,root_mean_squared_error\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8ede968c-c35b-40d8-a462-87b37ac2825c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('ToyotaCorolla - MLR.csv') # First i read the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "be0e4a02-aa49-40c7-9cf1-02cf609e200d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1436, 11)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape  #I chech its shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "45cba453-7c52-4281-a02e-c0c753eb686d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Price        0\n",
       "Age_08_04    0\n",
       "KM           0\n",
       "Fuel_Type    0\n",
       "HP           0\n",
       "Automatic    0\n",
       "cc           0\n",
       "Doors        0\n",
       "Cylinders    0\n",
       "Gears        0\n",
       "Weight       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum() # i check if any null values are present or not"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4542cc0a-0f15-4946-a87a-d4adbaa973bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()# i check duplicate values in dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "da06aa25-6380-4ce0-9093-95fe3d205098",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop_duplicates(inplace=True,ignore_index=True)# i remove it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "48e0e7c2-f24f-4c2e-9f13-e070313ff347",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Price</th>\n",
       "      <th>Age_08_04</th>\n",
       "      <th>KM</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>HP</th>\n",
       "      <th>Automatic</th>\n",
       "      <th>cc</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Cylinders</th>\n",
       "      <th>Gears</th>\n",
       "      <th>Weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>13500</td>\n",
       "      <td>23</td>\n",
       "      <td>46986</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13750</td>\n",
       "      <td>23</td>\n",
       "      <td>72937</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13950</td>\n",
       "      <td>24</td>\n",
       "      <td>41711</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14950</td>\n",
       "      <td>26</td>\n",
       "      <td>48000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13750</td>\n",
       "      <td>30</td>\n",
       "      <td>38500</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1170</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Price  Age_08_04     KM Fuel_Type  HP  Automatic    cc  Doors  Cylinders  \\\n",
       "0  13500         23  46986    Diesel  90          0  2000      3          4   \n",
       "1  13750         23  72937    Diesel  90          0  2000      3          4   \n",
       "2  13950         24  41711    Diesel  90          0  2000      3          4   \n",
       "3  14950         26  48000    Diesel  90          0  2000      3          4   \n",
       "4  13750         30  38500    Diesel  90          0  2000      3          4   \n",
       "\n",
       "   Gears  Weight  \n",
       "0      5    1165  \n",
       "1      5    1165  \n",
       "2      5    1165  \n",
       "3      5    1165  \n",
       "4      5    1170  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2b76db59-f01d-4c5b-8d12-cd6f4513f83b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Price         236\n",
       "Age_08_04      77\n",
       "KM           1263\n",
       "Fuel_Type       3\n",
       "HP             12\n",
       "Automatic       2\n",
       "cc             13\n",
       "Doors           4\n",
       "Cylinders       1\n",
       "Gears           4\n",
       "Weight         59\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique() # Then i check each column how many unique values "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "de5f43a2-8e13-47af-9d28-5741de052703",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Cylinders\n",
       "4    1435\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Cylinders.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "ec41d981-ea63-4450-b74f-3a931eb743d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gears\n",
       "5    1389\n",
       "6      43\n",
       "3       2\n",
       "4       1\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Gears.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "91de4344-f6e4-4ead-9711-4f4c421d8ef3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1435 entries, 0 to 1434\n",
      "Data columns (total 11 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   Price      1435 non-null   int64 \n",
      " 1   Age_08_04  1435 non-null   int64 \n",
      " 2   KM         1435 non-null   int64 \n",
      " 3   Fuel_Type  1435 non-null   object\n",
      " 4   HP         1435 non-null   int64 \n",
      " 5   Automatic  1435 non-null   int64 \n",
      " 6   cc         1435 non-null   int64 \n",
      " 7   Doors      1435 non-null   int64 \n",
      " 8   Cylinders  1435 non-null   int64 \n",
      " 9   Gears      1435 non-null   int64 \n",
      " 10  Weight     1435 non-null   int64 \n",
      "dtypes: int64(10), object(1)\n",
      "memory usage: 123.4+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()# i check th info of the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "58798c46-179b-45a0-a98c-c12737eef9a2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Price</th>\n",
       "      <th>Age_08_04</th>\n",
       "      <th>KM</th>\n",
       "      <th>HP</th>\n",
       "      <th>Automatic</th>\n",
       "      <th>cc</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Cylinders</th>\n",
       "      <th>Gears</th>\n",
       "      <th>Weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.0</td>\n",
       "      <td>1435.000000</td>\n",
       "      <td>1435.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>10720.915679</td>\n",
       "      <td>55.980488</td>\n",
       "      <td>68571.782578</td>\n",
       "      <td>101.491986</td>\n",
       "      <td>0.055749</td>\n",
       "      <td>1576.560976</td>\n",
       "      <td>4.032753</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.026481</td>\n",
       "      <td>1072.287108</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>3608.732978</td>\n",
       "      <td>18.563312</td>\n",
       "      <td>37491.094553</td>\n",
       "      <td>14.981408</td>\n",
       "      <td>0.229517</td>\n",
       "      <td>424.387533</td>\n",
       "      <td>0.952667</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.188575</td>\n",
       "      <td>52.251882</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4350.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>69.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1300.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.0</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>8450.000000</td>\n",
       "      <td>44.000000</td>\n",
       "      <td>43000.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1400.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1040.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>9900.000000</td>\n",
       "      <td>61.000000</td>\n",
       "      <td>63451.000000</td>\n",
       "      <td>110.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1600.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1070.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>11950.000000</td>\n",
       "      <td>70.000000</td>\n",
       "      <td>87041.500000</td>\n",
       "      <td>110.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1600.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>1085.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>32500.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>243000.000000</td>\n",
       "      <td>192.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>16000.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>4.0</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1615.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Price    Age_08_04             KM           HP    Automatic  \\\n",
       "count   1435.000000  1435.000000    1435.000000  1435.000000  1435.000000   \n",
       "mean   10720.915679    55.980488   68571.782578   101.491986     0.055749   \n",
       "std     3608.732978    18.563312   37491.094553    14.981408     0.229517   \n",
       "min     4350.000000     1.000000       1.000000    69.000000     0.000000   \n",
       "25%     8450.000000    44.000000   43000.000000    90.000000     0.000000   \n",
       "50%     9900.000000    61.000000   63451.000000   110.000000     0.000000   \n",
       "75%    11950.000000    70.000000   87041.500000   110.000000     0.000000   \n",
       "max    32500.000000    80.000000  243000.000000   192.000000     1.000000   \n",
       "\n",
       "                 cc        Doors  Cylinders        Gears       Weight  \n",
       "count   1435.000000  1435.000000     1435.0  1435.000000  1435.000000  \n",
       "mean    1576.560976     4.032753        4.0     5.026481  1072.287108  \n",
       "std      424.387533     0.952667        0.0     0.188575    52.251882  \n",
       "min     1300.000000     2.000000        4.0     3.000000  1000.000000  \n",
       "25%     1400.000000     3.000000        4.0     5.000000  1040.000000  \n",
       "50%     1600.000000     4.000000        4.0     5.000000  1070.000000  \n",
       "75%     1600.000000     5.000000        4.0     5.000000  1085.000000  \n",
       "max    16000.000000     5.000000        4.0     6.000000  1615.000000  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "25705512-bfe8-4cf2-8c84-716d9888d112",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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mRlWrFiBFStW3Lemffv22LRpU73L6dy5M2JiYuqt8fHxkf2iOBEREVE1/ugqERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKVaTg1BCQgJGjhwJJycnqFQqbN++XTZfCIG5c+fCyckJ5ubmGDx4MI4fPy6rKS0txfTp02FrawtLS0uMGjUKly9fltXk5eUhLCwMWq0WWq0WYWFhuHXrlqzm4sWLGDlyJCwtLWFra4sZM2agrKxMVnPs2DH4+fnB3Nwczs7O+OCDDyCEaGrbREREZICaHISKiorw2GOPYeXKlXXOX7hwIZYuXYqVK1fi8OHDcHR0REBAAG7fvi3VzJw5E9u2bcOWLVuwf/9+FBYWIiQkBJWVlVJNaGgo0tLSEBsbi9jYWKSlpSEsLEyaX1lZiREjRqCoqAj79+/Hli1bEBUVhfDwcKmmoKAAAQEBcHJywuHDh7FixQosXrwYS5cubWrbREREZICMm3qF4OBgBAcH1zlPCIHly5fjvffew9ixYwEAX331FRwcHPDNN9/g1VdfRX5+PtatW4eNGzdi6NChAIBNmzbBxcUFu3btQlBQEDIyMhAbG4uUlBT4+voCAD7//HP069cPmZmZ8PLyQlxcHE6cOIFLly7ByckJALBkyRJMmjQJ8+fPh42NDTZv3ow7d+5gw4YN0Gg08Pb2xqlTp7B06VLMmjUL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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.boxplot() # Then i plot the dataset whethere outlier is present or not\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "afe43850-43f8-4f02-a95a-8955149e2aba",
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy.stats.mstats import winsorize\n",
    "df['KM']=winsorize(df['KM'],limits=[0.05,0.05]) # We have two method to remove the outlier so i using winsorize method "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b1a84787-4053-494a-b4ad-e45d5991d519",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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3UXWq1cvjBgxAuHh4Vi+fDkA4LXXXkNwcDC8vb0BAIGBgejduzfCwsKwYMECXLt2DVFRUQgPD+cdY0RERNT0QSg2Nhb//Oc/MWXKFBQUFMDDwwOTJ0/G+++/L9XMmjUL5eXlmDJlCgoLCzFw4EAkJibCwcFBqlmyZAksLS0xbtw4lJeXY9iwYVizZg0sLCykmvXr1yMiIkK6uywkJARxcXHSfAsLC2zfvh1TpkzBkCFDYGNjg9DQUCxcuLCp2yYiIqI2qMmDkIODA5YuXSrdLl8XlUqF6OhoREdH37HG2toasbGxsg9irM3Z2Rnr1q2763g6d+6Mbdu21TdsIiIiUiB+1xgREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESkWgxAREREpFoMQERERKRaDEBERESmWZUsPgMjUVJbFyC3OQTtr+3prKysrcbnyMrKvZcPSsv6XR25xCVSWxU0xTCIiagEMQmT21A+k492DMY1a5pOETxqx/mEAnmnkqIiIqDVgECKzZ7g+EItGhaKbW8OOCB1IPoAhTw5p0BGhMwUliFh/pimGSURELYBBiMyeqHSEl6M3erto6601GAzItcxFL+deUKvV9dZX3yyCqLzaFMMkIqIWwIuliYiISLFMEoR+++03/PWvf4WLiwtsbW3xyCOPICMjQ5ovhEB0dDQ8PDxgY2ODoUOH4tixY7J16PV6TJs2Da6urrCzs0NISAguXbokqyksLERYWBi0Wi20Wi3CwsJw/fp1Wc2FCxcwevRo2NnZwdXVFREREaioqDBF20RERNTGNHkQKiwsxJAhQ6BWq/HDDz/g+PHjWLRoER544AGpZv78+Vi8eDHi4uJw6NAh6HQ6BAQE4MaNG1LNjBkzsGXLFsTHxyM5ORklJSUIDg5GVVWVVBMaGorMzEwkJCQgISEBmZmZCAsLk+ZXVVVh1KhRKC0tRXJyMuLj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i+nTp9e5r2q0toul7+WusfoEBwfLLoK93e3Pry1btkgfzxAcHCxWrFghu1j65s2b4s9//rN44IEH7un2+fp6rf36Wbx4sejQoYOwsbERQUFB4r///a9R76+//rpwcXG56+3zf/zxh3jllVeEi4uLsLa2Fj4+PmLbtm0N+tlVVVWJjz76SHTp0kWo1WrRuXNn6cL+c+fOiT//+c/C0dFR2NraigEDBoj09PQGrfdeTZgwQbqd2NLSUrRv314MHz5cfP7557ILz5vq9vktW7bIpv3rX/8SvXr1EjY2NsLZ2Vk8++yz4uzZs0KIW6+7qVOnii5duggrKyvRsWNHERISIrvVurq6WnTp0qXO52NDLpauPR6tVis9D4W4dTdkzXvkU089Jd2NePtzpr73yLou0l2xYoV45JFHhJ2dnXB0dBTDhg2T7rC9fPmyePPNN4WXl5dQq9XC3t5ePPHEE2LBggWitLRUCHHrwvtp06YJDw8PoVarhaenp3jppZekC9pzc3OFv7+/sLGxEZ6eniIuLq7en0d947qb3NxcAUD07NlTNv3ixYsCgOjWrZs0raKiQrz//vuia9euQq1WC51OJ5577jnx66+/CiHqvmnhww8/FK6ursLe3l68+uqrIiIiQgwaNEiaX9frv/Z7cWpqqnj44YeFRqNp9O3zKiEacKKUzMLcuXOxbNky2SFwIqLWqqysDB4eHvj8888xduzYlh4ONZOAgADodDqsXbu2WbbHi6XN2CeffILHH38cLi4uOHDgABYsWIA333yzpYdFRHRX1dXVyM/Px6JFi6DVahESEtLSQyITKSsrw7JlyxAUFAQLCwt8+eWX2Llzp+z6U1NjEDJjNedbr127hs6dOyMyMhKzZ89u0LL79+/HyJEj7zi/pKTkvscXExODmJiYOuc99dRT+OGHH+57G0TU9ly4cAFeXl7o1KkT1qxZA0tL/qoyVyqVCjt27MBHH30EvV4Pb29vbNq0CcOHD2++MfDUGNWlvLwcv/322x3n175l+15cu3btjncr2NjYyG7dJiIiMgUGISIiIlIs3j5PREREisUgRERERIrFIERERESKxSBEREREisUgRERERIrFIERERESKxSBEREREisUgRERERIr1/wG01ysoBkBAlwAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.boxplot()# after treat the outlier\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1c6c3049-4a89-4447-98bc-68f95de3d5b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Price</th>\n",
       "      <th>Age_08_04</th>\n",
       "      <th>KM</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>HP</th>\n",
       "      <th>Automatic</th>\n",
       "      <th>cc</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Cylinders</th>\n",
       "      <th>Gears</th>\n",
       "      <th>Weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>13500</td>\n",
       "      <td>23</td>\n",
       "      <td>46986</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13750</td>\n",
       "      <td>23</td>\n",
       "      <td>72937</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13950</td>\n",
       "      <td>24</td>\n",
       "      <td>41711</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14950</td>\n",
       "      <td>26</td>\n",
       "      <td>48000</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13750</td>\n",
       "      <td>30</td>\n",
       "      <td>38500</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1170</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Price  Age_08_04     KM Fuel_Type  HP  Automatic    cc  Doors  Cylinders  \\\n",
       "0  13500         23  46986    Diesel  90          0  2000      3          4   \n",
       "1  13750         23  72937    Diesel  90          0  2000      3          4   \n",
       "2  13950         24  41711    Diesel  90          0  2000      3          4   \n",
       "3  14950         26  48000    Diesel  90          0  2000      3          4   \n",
       "4  13750         30  38500    Diesel  90          0  2000      3          4   \n",
       "\n",
       "   Gears  Weight  \n",
       "0      5    1165  \n",
       "1      5    1165  \n",
       "2      5    1165  \n",
       "3      5    1165  \n",
       "4      5    1170  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "b39326c0-dd3a-4ab0-9359-2df6446ad3af",
   "metadata": {},
   "outputs": [],
   "source": [
    "Label =LabelEncoder()\n",
    "df['Fuel_Type']=Label.fit_transform(df['Fuel_Type'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "761c6739-6841-45ff-bb9c-e3370879817c",
   "metadata": {},
   "outputs": [],
   "source": [
    "target = df['Price']\n",
    "feature = df.drop('Price',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6401f5ae-fe30-4d15-b850-d5d618d0abe4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1148, 10)\n",
      "(287, 10)\n",
      "(1148,)\n",
      "(287,)\n"
     ]
    }
   ],
   "source": [
    "x_train,x_test,y_train,y_test = train_test_split(feature,target,train_size=0.80,random_state=100)\n",
    "print(x_train.shape)\n",
    "print(x_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "5a60ec1c-0b16-44ce-8e54-d2e9dfb275da",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-1.24985354e+02 -1.88370464e-02  4.87816457e+02  2.75454263e+01\n",
      "  3.98983397e+02 -4.85167748e-02  6.10259868e+00  7.61701813e-12\n",
      "  7.10295913e+02  1.91778418e+01]\n",
      "-8811.391524144386\n"
     ]
    }
   ],
   "source": [
    "mlr1 = LinearRegression()\n",
    "mlr1.fit(x_train,y_train)\n",
    "print(mlr1.coef_)\n",
    "print(mlr1.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "a32aff7b-be30-4be9-b13d-4dc36572e260",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df.corr(),annot=True,cmap='coolwarm')\n",
    "plt.show()\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "40da3e3d-6b0d-4651-85da-14f1d936af53",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature2 = df[['Age_08_04','KM','HP','Weight']]\n",
    "target2 = df[['Price']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "917df3f9-89ec-49c9-b49f-f62a0ff53dd2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1435, 4) (1435, 1)\n"
     ]
    }
   ],
   "source": [
    "print(feature2.shape,target2.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "585e3167-d4bb-42b5-aa75-e9f55ea1f018",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "16\n"
     ]
    }
   ],
   "source": [
    "print(feature2.duplicated().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1394277a-aac0-4009-a74b-ad4da548dc81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age_08_04</th>\n",
       "      <th>KM</th>\n",
       "      <th>HP</th>\n",
       "      <th>Weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>84</th>\n",
       "      <td>25</td>\n",
       "      <td>17051</td>\n",
       "      <td>97</td>\n",
       "      <td>1110</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>4</td>\n",
       "      <td>17051</td>\n",
       "      <td>116</td>\n",
       "      <td>1480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>4</td>\n",
       "      <td>17051</td>\n",
       "      <td>116</td>\n",
       "      <td>1480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>159</th>\n",
       "      <td>16</td>\n",
       "      <td>17051</td>\n",
       "      <td>110</td>\n",
       "      <td>1105</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>165</th>\n",
       "      <td>14</td>\n",
       "      <td>17051</td>\n",
       "      <td>110</td>\n",
       "      <td>1130</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Age_08_04     KM   HP  Weight\n",
       "84          25  17051   97    1110\n",
       "110          4  17051  116    1480\n",
       "111          4  17051  116    1480\n",
       "159         16  17051  110    1105\n",
       "165         14  17051  110    1130"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "feature2[feature2.duplicated()].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "39dd738d-8a67-4134-a2f5-5861d73c315d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1199\n"
     ]
    }
   ],
   "source": [
    "print(target2.duplicated().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9d2086d8-4195-427b-95c3-9b13f43d2451",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1435, 4) (1435, 1)\n"
     ]
    }
   ],
   "source": [
    "print(feature2.shape,target2.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "082072b8-d9d5-431d-9af3-dc24163ed864",
   "metadata": {},
   "outputs": [],
   "source": [
    "target2=target2.loc[feature2.index].reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "30579e84-e39c-4b61-bd94-3094ca426ef2",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature2.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "dce62841-7c0b-45cd-9cfb-d47d9fe26bec",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature2.drop_duplicates(inplace=True,ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "9d8583d0-bb9d-40d5-a3ad-232a73e3b662",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1135, 4)\n",
      "(284, 4)\n",
      "(1135, 1)\n",
      "(284, 1)\n"
     ]
    }
   ],
   "source": [
    "x_train2,x_test2,y_train2,y_test2 = train_test_split(feature2,target2,train_size=0.80,random_state=100)\n",
    "print(x_train2.shape)\n",
    "print(x_test2.shape)\n",
    "print(y_train2.shape)\n",
    "print(y_test2.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "a119ac4d-6a72-4a6a-83b6-03e370c6ec8f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-1.40651066e+02 -1.13587998e-02  1.08323835e+01  1.44696397e+01]]\n",
      "[2864.24885146]\n"
     ]
    }
   ],
   "source": [
    "mlr2 = LinearRegression()\n",
    "mlr2.fit(x_train2,y_train2)\n",
    "print(mlr2.coef_)\n",
    "print(mlr2.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "3fc75b03-ea6a-4b18-a132-dcee9645fe2e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Price</th>\n",
       "      <th>Age_08_04</th>\n",
       "      <th>KM</th>\n",
       "      <th>Fuel_Type</th>\n",
       "      <th>HP</th>\n",
       "      <th>Automatic</th>\n",
       "      <th>cc</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Cylinders</th>\n",
       "      <th>Gears</th>\n",
       "      <th>Weight</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>13500</td>\n",
       "      <td>23</td>\n",
       "      <td>46986</td>\n",
       "      <td>1</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13750</td>\n",
       "      <td>23</td>\n",
       "      <td>72937</td>\n",
       "      <td>1</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13950</td>\n",
       "      <td>24</td>\n",
       "      <td>41711</td>\n",
       "      <td>1</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14950</td>\n",
       "      <td>26</td>\n",
       "      <td>48000</td>\n",
       "      <td>1</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1165</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13750</td>\n",
       "      <td>30</td>\n",
       "      <td>38500</td>\n",
       "      <td>1</td>\n",
       "      <td>90</td>\n",
       "      <td>0</td>\n",
       "      <td>2000</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1170</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Price  Age_08_04     KM  Fuel_Type  HP  Automatic    cc  Doors  Cylinders  \\\n",
       "0  13500         23  46986          1  90          0  2000      3          4   \n",
       "1  13750         23  72937          1  90          0  2000      3          4   \n",
       "2  13950         24  41711          1  90          0  2000      3          4   \n",
       "3  14950         26  48000          1  90          0  2000      3          4   \n",
       "4  13750         30  38500          1  90          0  2000      3          4   \n",
       "\n",
       "   Gears  Weight  \n",
       "0      5    1165  \n",
       "1      5    1165  \n",
       "2      5    1165  \n",
       "3      5    1165  \n",
       "4      5    1170  "
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "8ee5c9e2-22d3-4273-99e4-2a6dd5c8ccf7",
   "metadata": {},
   "outputs": [],
   "source": [
    "scale = StandardScaler()\n",
    "df[['Age_08_04','KM','Fuel_Type','HP','Automatic','cc','Doors','Cylinders','Gears','Weight']]=scale.fit_transform(df[['Age_08_04','KM','Fuel_Type','HP','Automatic','cc','Doors','Cylinders','Gears','Weight']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c3ce4c42-18c1-4c5a-b653-9e862604aeb3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1435, 11)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "c358bd34-a5a1-46ae-892e-80769a73ce02",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a6889fdf-919f-4178-8e7b-881907f8b0ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop_duplicates(inplace=True,ignore_index=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "122a7e28-1c18-481d-b8d7-c6d72bf5614a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "from statsmodels.tools.tools import add_constant"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "414536ee-3c90-48e7-9d75-fe9696c07af7",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df[['Age_08_04','KM','HP','Fuel_Type','Automatic','cc','Doors','Gears','Weight']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "2dff1172-43fc-4256-8dd7-e0bc2da6b994",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Feature       vif\n",
      "0      const  1.000013\n",
      "1  Age_08_04  1.953785\n",
      "2         KM  1.916671\n",
      "3         HP  1.480339\n",
      "4  Fuel_Type  2.328557\n",
      "5  Automatic  1.063732\n",
      "6         cc  1.166876\n",
      "7      Doors  1.185931\n",
      "8      Gears  1.113515\n",
      "9     Weight  2.302905\n"
     ]
    }
   ],
   "source": [
    "x = add_constant(x)\n",
    "vif_data = pd.DataFrame()\n",
    "vif_data ['Feature']=x.columns\n",
    "vif_data['vif']=[variance_inflation_factor(x.values, i)for i in range(x.shape[1])]\n",
    "print(vif_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "10442f70-60d8-426f-a745-5f0da54bb379",
   "metadata": {},
   "outputs": [],
   "source": [
    "target3 = df[['Price']]\n",
    "feature3 = df.drop('Price',axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "9e3ab48f-9e81-457b-ab84-294b75e0904e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1433, 1) (1433, 10)\n"
     ]
    }
   ],
   "source": [
    "print(target3.shape,feature3.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "be430200-3876-4591-b463-8f8bb80f6f91",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1074, 10)\n",
      "(359, 10)\n",
      "(1074, 1)\n",
      "(359, 1)\n"
     ]
    }
   ],
   "source": [
    "x_train3,x_test3,y_train3,y_test3 = train_test_split(feature3,target3,train_size=0.75,random_state=100)\n",
    "print(x_train3.shape)\n",
    "print(x_test3.shape)\n",
    "print(y_train3.shape)\n",
    "print(y_test3.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "15371113-9733-41ef-8aa6-2f00d0227b52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-2.33490042e+03 -5.69500100e+02  2.28800307e+02  4.23862878e+02\n",
      "   7.55166758e+01 -1.17472796e+01  5.24186765e+00 -2.27373675e-13\n",
      "   1.06768047e+02  9.64009548e+02]]\n",
      "[10721.58612162]\n"
     ]
    }
   ],
   "source": [
    "mlr3 = LinearRegression()\n",
    "mlr3.fit(x_train3,y_train3)\n",
    "print(mlr3.coef_)\n",
    "print(mlr3.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "f3e2b15f-da23-41be-8d95-050c0713d03b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "f28b8030-35b5-4e8c-b45e-2f4a4e10a7ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pre1=mlr1.predict(x_test)\n",
    "r2 =r2_score(y_pre1,y_test)\n",
    "mse = mean_squared_error(y_pre1,y_test)\n",
    "rmse = np.sqrt(mse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "1f300f76-397c-4c48-9d2d-15d66bc65d29",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pre2=mlr2.predict(x_test2)\n",
    "r2_2=r2_score(y_pre2,y_test2)\n",
    "mse2 = mean_squared_error(y_pre2,y_test2)\n",
    "rmse2 = np.sqrt(mse2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "e30e6cff-3841-4582-bce5-f589245be9d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pre3=mlr3.predict(x_test3)\n",
    "r2_3 =r2_score(y_pre3,y_test3)\n",
    "mse3 = mean_squared_error(y_pre3,y_test3)\n",
    "rmse3 = np.sqrt(mse3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "0f31f77e-9a40-4c39-be1d-7057316dc09b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "r2 Persentage Score : 0.8409894986639668\n",
      "Mean Square Error Score : 1613329.9042350468\n",
      "Root Mean Square Error Score : 1270.169242359083\n"
     ]
    }
   ],
   "source": [
    "print('r2 Persentage Score :',r2)\n",
    "print('Mean Square Error Score :',mse)\n",
    "print('Root Mean Square Error Score :',rmse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "b615a104-3068-4664-92bc-20e338d5f6d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "r2 Persentage Score : 0.7099895611892455\n",
      "Mean Square Error Score : 3119990.0873704646\n",
      "Root Mean Square Error Score : 1766.3493673026478\n"
     ]
    }
   ],
   "source": [
    "print('r2 Persentage Score :',r2_2)\n",
    "print('Mean Square Error Score :',mse2)\n",
    "print('Root Mean Square Error Score :',rmse2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "826ea322-d689-43db-83e8-8eb77f47da14",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "r2 Persentage Score : 0.8298084414468994\n",
      "Mean Square Error Score : 1748966.4231701288\n",
      "Root Mean Square Error Score : 1322.4849425116827\n"
     ]
    }
   ],
   "source": [
    "print('r2 Persentage Score :',r2_3)\n",
    "print('Mean Square Error Score :',mse3)\n",
    "print('Root Mean Square Error Score :',rmse3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "164ff961-9e08-496c-b688-85cf90666c6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "rigi = Ridge(alpha=1.0)\n",
    "rigi.fit(x_train,y_train)\n",
    "y_predict = rigi.predict(x_test)\n",
    "rig=r2_score(y_predict,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "c443e8ed-69a8-4f00-8601-d8837746cb7b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8409986390714904"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "50daeb29-d934-48ee-a3e0-aa56d501079f",
   "metadata": {},
   "outputs": [],
   "source": [
    "rigi = Ridge(alpha=1.0)\n",
    "rigi.fit(x_train2,y_train2)\n",
    "y_predict2 = rigi.predict(x_test2)\n",
    "rig_2=r2_score(y_predict2,y_test2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "6dfd5e98-aaa4-4727-a724-89824da8249c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.709988628417038"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rig_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "6497698c-3af1-49e9-b35d-f7737415b919",
   "metadata": {},
   "outputs": [],
   "source": [
    "rigi = Ridge(alpha=1.0)\n",
    "rigi.fit(x_train3,y_train3)\n",
    "y_predict3 = rigi.predict(x_test3)\n",
    "rig_3=r2_score(y_predict3,y_test3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "a75f782f-ef73-4194-9145-6aed05444633",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8295574123241148"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rig_3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "7fa0960b-d7f0-4361-a865-cffeb72f8621",
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso = Lasso(alpha=0.1)\n",
    "lasso.fit(x_train,y_train)\n",
    "y_predi = lasso.predict(x_test)\n",
    "laso=r2_score(y_test,y_predi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "69bc3c32-e7f9-4219-b7d1-0c137463cae0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8648168402287613"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "laso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "d81a3e74-3a58-40a0-88ab-1fdea2805379",
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso = Lasso(alpha=0.1)\n",
    "lasso.fit(x_train2,y_train2)\n",
    "y_predi2= lasso.predict(x_test2)\n",
    "laso_2=r2_score(y_test2,y_predi2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "932a08e0-da8f-48f5-b172-4d6365d09c09",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7447707920555899"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "laso_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "29c30db1-75c6-4067-a0fc-857b23d74a94",
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso = Lasso(alpha=0.1)\n",
    "lasso.fit(x_train3,y_train3)\n",
    "y_predi3 = lasso.predict(x_test3)\n",
    "laso3=r2_score(y_test3,y_predi3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "103e58f7-592e-45a4-bf1f-547ac7101eec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8640521850751106"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "laso3"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d2e2c77-ef92-45bf-8b39-bf347f485109",
   "metadata": {},
   "source": [
    "Normalization(minmax scaling) rescales data b/w 0 to 1\n",
    "Standardization (Z-score Scaling) transform the data and is ranges b/w -3 to 3\n",
    "Both techinque were used for scaling the data because is very useful to ML model\n",
    "Then scaling method only used for distance based alogritm and gradient based alogrthim\n",
    "\n",
    "Well mulicollinearty is the techinque is used to check the two independent variables are highly correlated or not\n",
    "If the variables are highly correleated means the model will be unstable and is also affect the model also\n",
    "We have a lot of technique to remove correlated data :\n",
    "VIF is the one type of techinque to identify which variable have highly correleated\n",
    "Drop variable with VIF>5to10\n",
    "Another one drop or combine them etc.."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61b56151-7114-40f6-b275-0fa5862b2678",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
