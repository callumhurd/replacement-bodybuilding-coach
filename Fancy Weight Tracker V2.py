import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy import stats

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")






# Define the starting date
start_date = datetime(2024, 8, 16)
target_date = pd.Timestamp(2025,2,1)


# Data (Cleaned in one go within the function)
weights_list = [
    62.0, 63.0, 64.8, 67.4, 66.9, 66.6, 67.2, 67.2, 67.4, 67.8, 68.0, 68.3, 68.7, 67.9, 69.3, 69.5, 69.8, 70.2,
    69.5, 68.2, 70.2, 70.4, 70.5, 70.6, 70.8, 69.9, 69.8, 69.4, 70.1, 70.2, 70.6, 70.7, 70.4, 71.4, 69.7, 70.5,
    70.6, 71.2, 70.4, 70.5, 71.5, 71.7, 69.3, 71.3, 71.5, 71.2, 72.2, 71.7, 71.7, 72.7, 71.9, 71.8,
    72.7, 72.6, 72.5, 72.4, 72.1, 71.9, 71.9, 72.6, 72.1, 71.5,71.5,73.6,72.9,
    71.8,71.9,71.8,72.0,72.2,72.9,74.3,73.6,74.1,74.0,73.1,75.0,74.5,74.0,73.4,73.3,
    72.8,74.2,74.2,73.5,74.4,73.7,74.3,74.6,74.6,
]

print(len(weights_list))

carb_intensity_list = [
    "Carb up", "Carb up", "Cheat", "Med", "Low", "Low", "Med", "Low", "Low", "High", "High", "High", 
    "Low", "Low", "Med", "Low", "Low", "Med", "Low", "Med", "Low", "High", "High", "Med", 
    "Low", "Low", "Low", "Med", "Low", "High", "Low", "Med", "Low", "Low", "Low", "Med", 
    "High", "Low",'Med','Med','Med','Med','Med','Med','Med','Med','Med',"Med","Med",
    "Med","Med","Med","Med","Med","Med","Med","Med","Med","Med","Med",'Med','Med'
    ,'Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med'
    ,'Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med','Med'
]

print(len(carb_intensity_list))

comments_list = [
    "Lots of energy", "Lots of energy", "Lots of energy", "Full of water", "Full of water", "Lots of energy", 
    "Bloating full of water", "Full of water", "Full of water", "Full of water", "Extreme cravings", 
    "Extreme hunger", "Extreme hunger", "Slightly less cravings", "Energy stabilizing", "Feeling better", 
    "Hunger slightly reducing", "Feeling normal", "Moderate hunger", "Good energy", "Less bloating", 
    "Normal cravings", "Slight energy dip", "Feeling stable", "Stable energy", "Gradual energy increase", 
    "Feeling good", "Feeling full", "Energy normalizing", "Less hunger", "Feeling great", 
    "Stable hunger", "Energy levels balanced", "Feeling consistent", "No major hunger", 
    "High energy day", "Slight tiredness", "Feeling steady", "really good", "really good", "really good", "really good", "really good",
    "really good", "really good", "really good", "really good", "hungry", "Increased water retention", "Energy levels rising",
    "Feeling a bit bloated", "Water weight noticeable", "Stable energy throughout the day", "Slight dip in energy", 
    "Higher energy in the afternoon",  "Water retention stabilizing",  "Energy slowly improving", "Feeling more hydrated", 
    "Energy surges after meals", "Water retention increasing", "No energy dips", "Energy becoming more stable"
    'strength through the roof','looking very full', 'muscle looking dense', 'shit session today', 'full of water'
    ,'strength through the roof','strength through the roof','strength through the roof','strength through the roof'
    ,'strength through the roof','strength through the roof','strength through the roof','strength through the roof'
    ,'strength through the roof','strength through the roof','strength through the roof','strength through the roof'
    ,'strength through the roof','strength through the roof','strength through the roof','full of water'
    ,'full of water','full of water','full of water','full of water','full of water','full of water'
    ,'full of water','full of water'
]

print(len(comments_list))

calories_list = [
    2629, 3500, 5000, 2675, 2473, 2406, 2612, 2624, 2688, 5811, 5000, 4900, 4700, 2001, 2235, 3100, 2100, 2312,
    2000, 3200, 2300, 2300, 2300, 2300, 2300, 2300, 4000, 3800, 2400, 2300, 2450, 2500, 3000, 2600, 2700, 2800,
    2400, 2550,2994,3608,2408,1806,3219,2813,3137,3000,4000,2400,2994,
    3604,2408,1806,3219,2813,3137,4000,3458,2300,2853,2909,2699,2614,
    3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000
    ,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000,3000
]

print(len(calories_list))

## sentiment analysis function roBERTa 
def analyze_sentiments(comments):
    inputs = tokenizer(comments, return_tensors="pt", padding=True, truncation=True)

    # Pass the inputs through the model
    outputs = model(**inputs)

    # logits are unnormalised outputs and then you apply softmax or sigmoid to sort it out
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class indices
    predicted_classes = torch.argmax(probabilities, dim=-1)
    
    # everythings in 0,1,2 so we need a map
    labels = ["Negative", "Neutral", "Positive"]
    predicted_labels = [labels[class_idx.item()] for class_idx in predicted_classes]

    # Count the occurrences of each sentiment
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for label in predicted_labels:
        sentiment_counts[label] += 1


 #   for comment, label, probs in zip(comments, predicted_labels, probabilities):
  #      print(f"Comment: {comment}")
   #     print(f"Sentiment: {label}")
    #    print(f"Probabilities: {probs.tolist()}")
     #   print("-" * 50)

    return [sentiment_counts, predicted_labels]


result = analyze_sentiments(comments_list)
print("Sentiment Counts:", result)


def Analyse_Data(weights_list, carb_intensity_list, comments_list, calories_list, start_date, goal, goal_weight, target_date):
    # Prepare lists for DataFrame columns
    dates = [start_date + timedelta(days=i) for i in range(len(weights_list))]
    
    # Create the DataFrame
    df = pd.DataFrame({
        'index': list(range(len(weights_list))),
        "date": dates,
        "weights": weights_list,
        "carb intensity": carb_intensity_list,
        "comments": comments_list,
        "Calories": calories_list
    })
    
    ## Define a target log model for whatever the goal weight is
    start_weight = weights_list[0]
    total_days = (target_date - start_date).days
    a = (goal_weight - start_weight) / np.log(total_days + 1)
    b = start_weight
    
    def log_model(x):
            return a * np.log(x + 1) + b
    
    df['target_weights'] = log_model(df['index'])
    
    # RMSE test
    observed = df['weights']
    expected = df['target_weights']
    rmse = np.sqrt(np.mean((observed - expected)**2))
    
    mean_weight = np.mean(observed)
    rmse_threshold = 0.1 * mean_weight  # 10% of mean weight
    
    # Result based on RMSE threshold
    rmse_result = "Fail to reject" if rmse < rmse_threshold else "Reject"
    
    # Sentiment analysis bit
    results = analyze_sentiments(list(df['comments']))
    sentiment_result = results[0]
    lst_of_labels = results[1]
    
    if len(lst_of_labels) > 14:
        last2weekslst = lst_of_labels[-14]
    else:
        last2weekslst = lst_of_labels
    
    for i in range(len(last2weekslst)-3):
        if last2weekslst[i] == "Negative" and last2weekslst[i+1] == "Negative" and last2weekslst[i] == "Negative":
            comment_on_sentiment = "Looks like things over the last few days have been suboptimal, has nutrition been on point? you've had some negative streaks in this period. Take a deload if this is related to recovery."
        else:
            comment_on_sentiment = "Looks good buddy diet's been on point, youve had consistently good periods of time with no negative streaks, keep going the way you're going."
    
    # Calculate moving averages
    df["5_day_moving_avg"] = df["weights"].rolling(window=5, min_periods=1).mean()
    df["3_day_moving_avg"] = df["weights"].rolling(window=3, min_periods=1).mean()
    avg_calories = df['Calories'].mean()

    # Initial plot
    plt.figure(figsize=(14, 7))
    plt.plot(df['index'], df['weights'], color='black', label='Daily Weights', marker='o')
    plt.plot(df['index'], df['5_day_moving_avg'], color='blue', label='5-Day Moving Average')
    plt.plot(df['index'], df['3_day_moving_avg'], color='green', label='3-Day Moving Average')
    plt.axhline(y=goal_weight, color='red', linestyle='--', label='Goal weight')
    plt.title(f"Weight Analysis with Moving Averages (Goal: {goal})")
    plt.xlabel("Index (Days)")
    plt.ylabel("Weight")
    plt.legend()

    # Weight change calculations
    weight_loss_abs = abs(df['weights'].iloc[2] - df['weights'].iloc[-1])
    theor_weight_loss = df['5_day_moving_avg'].iloc[5] - df['5_day_moving_avg'].iloc[-1]

    # Linear regression
    x = np.array(df['index']).reshape((-1, 1))
    y = np.array(df['weights'])
    model = LinearRegression().fit(x, y)
    
    # Prediction (linear regression)
    last_index = df['index'].iloc[-1]
    array_of_goals = np.array([last_index + 14]).reshape(-1, 1)
    y_pred = model.predict(array_of_goals)

    # Logarithmic regression
    x_log = np.log(x + 1)  # +1 to prevent log(0) error
    model_log = LinearRegression().fit(x_log, y)
    y_pred_log = model_log.predict(np.log(array_of_goals + 1))

    # Days left to goal weight (linear model)
    days_left = (goal_weight - model.intercept_) / model.coef_[0] - last_index

    # Plot regression lines
    plt.plot(x, model.coef_[0] * x + model.intercept_, color='red', label='Linear Regression')
    plt.plot(x, model_log.predict(x_log), color='orange', label='Logarithmic Regression')
    
    # Week-on-week weight change
    df['week'] = df['index'] // 7
    weekly_avg_weights = df.groupby('week')['weights'].mean()
    week_on_week_change = weekly_avg_weights.diff().dropna().tolist()

    # Goal-specific status
    rate_of_change = 7 * model.coef_[0]
    if goal == 'bulk' and rate_of_change >= 0.1:
        current_status = "on track"
    elif goal == 'cut' and rate_of_change <= -0.25:
        current_status = "on track"
    else:
        current_status = "not on track"


    # Prepare the data for DataFrame
    result = {
        'Metric': [
            'Final Weight',
            'Current Moving Avg',
            'Weight Loss (Absolute)',
            'Moving Avg Weight Change',
            'Model Prediction (14 Days)',
            'Logarithmic Prediction (14 Days)',
            'Rate of Change',
            'Days Left to Goal',
            'Average Calories',
            'Status (Goal)',
            'Weekly Changes (Wk-on-Wk)',
            'RMSE Value',
            'RMSE Threshold',
            'RMSE Test Result',
            'Number of Good and Bad Days',
            'Comment on Sentiment'
        ],
        'Value': [
            df['weights'].iloc[-1],                             
            df["5_day_moving_avg"].iloc[-1],                     
            weight_loss_abs,                                     
            abs(theor_weight_loss),                              
            y_pred[0],                                           
            y_pred_log[0],                                       
            rate_of_change,                                      
            days_left,                                           
            avg_calories,                                        
            current_status,                                      
            week_on_week_change,                                 
            rmse,                                                
            rmse_threshold,                                      
            rmse_result,                                         
            sentiment_result,                                    
            comment_on_sentiment                                 
        ]
    }
    
    df_result = pd.DataFrame(result)
    
    


    return df_result

# Example function call
goal = "bulk"
goal_weight = 75
result = Analyse_Data(weights_list, carb_intensity_list, comments_list, calories_list, start_date, goal, goal_weight, target_date)
print(result)









