First of all, I want to thank the host for organizing this competition. Despite any challenges, I found the dataset rich with opportunities for exploration, feature engineering, and learning. I truly enjoyed participating.

A few words on the metric's hackability: it was a significant issue, but not the only one. The linear time trend penalty included in the metric can be substantial when there's no actual predictable trend in the ginis. I discussed this issue in more detail [here](https://www.kaggle.com/code/eivolkova/stability-metric-issue-probabilistic-approach/notebook). I strongly believe that the downward slope observed in the test set is related to changing market conditions rather than the deterioration of the model's prediction quality. These changes in market conditions create a stochastic trend, which is almost always observed in non-differentiated financial time series and is not predictable. Therefore, extrapolating this downward slope is not valid.

### CV strategy
In this competition, it was incredibly difficult to match CV and LB scores. I tried various approaches: simple cross-validation, time series CV, time series CV with a gap, and using only hold-out samples from weeks 80, 65, 50... The problem was that nothing correlated with the LB, and there was no negative slope to estimate the model's stability. My final approach was to split the data into five folds sequentially based on weeks, and drop slope term from the metric as it didn't have any meaning withing this CV scheme.

#### Finding a slope
I also tried a fancy approach to replicate the downward slope. I set aside the last 40 weeks (50-91) as a holdout set and trained two models: one on the first 25 weeks (0-25) and the second on the next 25 weeks (25-50). This was to estimate how much the model performance drops when trained on the most recent data compared to older data. Then I concatenated predictions from the second model with predictions from the first model and calculated the ginis stability metric, which now had a negative slope penalty. However, this approach also didn't correlate with LB scores, so I only considered it for feature selection purposes for a few models.

### Feature engineering
- First, I reviewed the data and all four hundred features provided, attempting to create meaningful features based on them (ratios, differences, etc.) and aggregations.
- **Person Data**: I noticed that when num_group1=0, the data pertains to the applicant, while other values correspond to related persons. Thus, I extracted data with num_group1=0 and added it to the feature set, applying aggregations for the rest.
- **Credit Bureau Data**: I noticed that there are duplicate columns for active and closed contracts, so I merged each pair into one to build aggregations based on all contracts, retaining active and closed contract aggregations as well.
- **Time Aggregations**: For credit bureau data, I created aggregations over different time intervals. For contracts data, I used contracts ending in the last 3, 5, or 7 years. For installment data, I used 1 month, 6 months, and 1, 2, or 3 years. This helped to improve the CV score significantly. I haven't tried aggregations based on the start of the credit, maybe it could yield even better results.
- **Categorical Features Aggregations**: I used mode, unique values count, and a measure of diversity (number of unique values divided by the total number per each case_id).
- **Same Data from Different Source**: I concatenated birth date, employment length, taxes, and other columns that were present in the dataset more than once. This improved CV scores; however, removing the original columns reduced CV scores. A better strategy might be to calculate the mode from all sources.
- **Credit Bureau Sources Concatenation**: As there are two sources for previous contracts data (a and b), I manually concatenated them. However, this didn't significantly improve results, so I kept this only in one model.
- **DPD features**: For previous applications and credit bureau data I created aggregations only for those rows where DPD is different from zero.

## Feature selection
For feature selection, I used two approaches: forward feature selection and null feature importances (borrowed from [previous competition notebook](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances)).

## Ensembling
When I started ensembling models, the correlation between CV and LB improved. It wasn't perfect, but ensembling finally allowed me to achieve LB scores higher than 0.571. I realized that training a few different tree-based models like LightGBM, CatBoost, and XGBoost wasn't enough. Initially, I used around 300 features selected by forward feature selection and added random subsets of features with different parameters for LightGBM and various seeds. Then I understood that to achieve even greater diversity, I needed to repeat the feature engineering process from scratch, create a new dataset, train a model on it, and add it to the existing ensemble. This resulted in having six different data processors, significantly enhancing the diversity of the solutions. In the end, I used an ensemble which gave me a good CV score and the highest LB score I could achieve. 

## Models
I used only LightGBM and CatBoost. I tried to build a few simple DL models, however, they didn't add anything to my ensemble, so I dropped them.

## Hack

I restored date information using the "refreshdate_3813885D" column from the credit_bureau_a1* files. The minimum value per case_id is almost always 3/1/2019. The difference between "refreshdate_3813885D" and "date_decision" has a near-perfect correlation with "date_decision". Since date differences are preserved in the test set, I restored the original date by subtracting this difference from 3/1/2019.

The percentage of correctly restored dates in the training dataset was 87%, with 3% errors and 10% missing values.

This method helped me achieve a 0.655 score on the public LB (with a hack). However, the publicly shared method worked better, giving me a 0.666 score on the public LB. The discrepancy might be due to false positives in the second half of the test sample. 

## What didn't work
- **Submodels**: I attempted to follow past Home Credit competition winners' approach by creating submodels for each dataset with depth >0. For each dataset I added 'target' column from train_base, trained models on CV, and used different aggregations of OOF predictions as meta-features (see this [discussion](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64596)). This only worked for credit bureau data on CV, and it significantly increased time and RAM requirements, so I abandoned this approach.
- **Scaling**: I was concerned about the non-stationarity of many features, primarily due to inflation for amount columns. Scaling by inflation measures worsened CV. Scaling all amount columns by income didn't work as well, even when I used scaled features in addition to original ones. This might be due to unreliable data on income: I noticed low correlation between income data from person1 and static0 sources (see this [discussion](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/491927)). This might be due to data provider post-processing or applicant mistakes.
- **Other approaches** that didn't work: average target for N closest neighbors; pseudo-labeling; augmenting the training dataset with data from previous applications; creating dummies for categorical features from depth>0 files and summing up; weightning observations with higher weight on recent ones; excluding features based on adversarial validation; weighted average of depth>0 columns based on time; creating seperate models for the case_ids with credit history and without it.

Link to Kaggle discussion. #TODO