import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt




X = np.loadtxt('./data_cat/CA-Mcat1.csv')
X = X[:, :2500]
y = np.loadtxt('./data/meander_label.csv')


# X = np.loadtxt('./data_cat/cat_meander.csv')
# X = X[:, :2500]
# y = np.loadtxt('./data_cat/meander_label.csv')

# X, y = make_moons(n_samples=264, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#集成投票器
clf_lr = LogisticRegression(penalty='l2', C=1000, random_state=0)
clf_dt = DecisionTreeClassifier(random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors=1, p=2)
pipe1 = make_pipeline(StandardScaler(), clf_lr)
pipe2 = make_pipeline(StandardScaler(), clf_dt)
pipe3 = make_pipeline(StandardScaler(), clf_knn)

models = [('lr', pipe1),
          ('dt', pipe2),
          ('KNN', pipe3)]

ensembel = VotingClassifier(estimators=models, voting='soft')


#分类结果（训练集）
all_model = [pipe1, pipe2, pipe3, ensembel]
clf_labels = ['LogisticRegression', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'Ensemble']
for clf, label in zip(all_model, clf_labels):
              score = cross_val_score(estimator=clf,
                                     X=X_train,
                                     y=y_train,
                                     cv=10,
                                     scoring='roc_auc')
              print('roc_auc:%0.2f(+/-%0.2f)[%s]' % (score.mean(), score.std(), label))




#不用模型的acc_roc曲线（测试集）
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
plt.figure(figsize=(6.4, 4.8))
font = {'family': 'Times New Roman',
        'size': 12,
        }
# sns.set(font_scale=1.2)
plt.rc('font', family='Times New Roman')
for clf, label, clr, ls in zip(all_model, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.3f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
plt.legend(loc='lower right', fontsize=12)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
# plt.grid()
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('ROC for Machine Learning', fontsize=16)
plt.savefig('./vote-roc.png', format='png')
plt.show()







