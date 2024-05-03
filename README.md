# iremSezerNotebook

# Python

## List comprehensions

```
loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]
```

## Underfitting & Overfitting

![uo](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/29de5db8-3d00-4b2a-b998-5a7a7fe43a12)

Model eğitim verilerinden elde edilen örüntülere göre oluşturulur. Bu işlem sonucunda iki şeyden biri olabilir; model aşırı öğrenebilir veya eksik öğrenebilir. Bu durumda model yeterli öngörüde bulunamayacak ve tahminlerimizde hata oranı yüksek olacaktır.


## Underfitting

Modelin verilerdeki temel örüntüleri yakalamak için çok basit olması ve bu nedenle kötü performans göstermesi 

## Overfitting

Model, eğitim için kullanılan veri seti üzerinde gereğinden fazla çalışıp ezber yapmaya başlamışsa ya da eğitim seti tek düze ise overfitting olma riski büyük demektir.

![uo.PNG](attachment:uo.PNG)

# Pandas

## iloc & loc

Loc komutu ile etiket kullananarak verilere ulaşırken, iloc komutunda satır ve sütün index numarası ile verilere ulaşır, Yani loc komutunu kullanırken satır yada kolon ismi belirtirken, iloc komutunda satır yada sütünün index numarasını belirtmez.

# Feature Engineering 

![1666105956546](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/b819196a-316e-4802-a2ac-b0b190d33901)

Feature engineering (Özellik Mühendisliği), ham verileri seçme, değiştirme ve denetimli öğrenmede kullanılabilecek özelliklere dönüştürme ve veriyi makine öğrenmesi modellerine hazırlama sürecidir. 

Makine öğrenimi algoritmalarının daha iyi performans göstermesine yardımcı olan özellikler veya girdi değişkenleri oluşturmak için alan bilgisini kullanma sürecidir.

Modelin tahmin gücünü arttırmaya yardımcı olur.

## Mutual Information
Ortaklı Bilgi Yöntemi

Bilgi teorisi alanından gelen ortaklı bilgi (mutual information), bilgi kazanımının özellik seçimine uygulanmasıdır.

Ortaklı bilgi iki değişken arasında hesaplanır ve diğer değişkenin bilinen bir değeri verildiğinde bir değişken için belirsizlikteki azalmayı ölçer.

```
#import the required functions and object.
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

#Kalmasını istediğiniz değişken sayısı
select_k = 10

#Değişkenlerin seçim stratejisi belirlenir
#mutual_info_classif = ortalı bilgi yönteminin kullanılmasıdır
selection = SelectKBest(mutual_info_classif, k=select_k).fit(x_train, y_train)

#İlişkili olan değişkenler gösterilir.
features = x_train.columns[selection.get_support()]
print(features)

from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores
```

## Outlier Detection
Aykırı Gözlem


![not22](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/6cb3dd3c-0802-4e52-9536-395c81fdf27c)

## K-Means Algoritması

![k1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/dbd81a8d-94f8-4d61-b5c4-e1b370f4b821)

Benzer nesneleri otomatik olarak aynı gruplara gruplandırır.

Farklı kümelerdeki veri noktaları çok farklıyken, aynı alt gruptaki (küme) veri noktalarının çok benzer olması nedeniyle verilerdeki alt grupların belirlenmesi görevi olarak tanımlanabilir. 

Unsupervised learning (gözetimsiz öğrenme) ve kümeleme algoritmasıdır.

```
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()
sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);
```


![k2](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/aa4c4326-48b9-4a25-a42f-26e0e16d4b11)

## Deep Leaarning

Derin Öğrenme bir makine öğrenme yöntemidir. 

Verilen bir veri kümesi ile çıktıları tahmin edecek yapay zekanın eğitilmesine olanak sağlar. 

Yapay zekayı eğitmek için hem denetimli hem de denetimsiz öğrenme kullanılabilir.

## Data Cleaning

### Scaling

Scaling (ölçeklendirme) verilerin aralığını değiştirir.


```python
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()
```


![s1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/4439facc-867d-48f1-9701-ade0d09c1956)

### Normalization (Normalleştirme) 

Verilerin dağıtım şeklini değiştirir.

```python
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()
```

![normalized_data](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/825475f0-52d2-4051-b527-58f4eb1160be)

### Embedding : Bir dilin veya verilen verideki kelimelerin tek tek, daha az boyutlu bir uzayda gerçek değerli vektörler olarak ifade edilmesidir.

Bir nesneyi başka bir nesnenin içine gömmek, yerleştirmek.

![Embedding](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*SYiW1MUZul1NvL1kc1RxwQ.png)

---

# LLM Bootcamp

- Deploying an MVP
  
![llm1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/c6d6b628-4384-4487-a848-e28d29d70214)

-
![llm2](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/8e780dce-d81d-43d6-9bf4-013649d901d3)

-
![llm3](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/db4b9412-b616-4063-9159-17c83556ef2b)

-
![llm4](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/11f23374-376e-4a76-80ca-9e5eb89683ad)

## Zero Shot

![Zero Shot](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/4031544d-bc1c-4d7b-9679-6065d74b1a08)

## One-hot Encoding

![One-hot Encoding](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/707300af-61b3-4631-b1d5-5bef9b86f52f)

## Few-Shot Learning

![image](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/1a0870be-b41c-4d15-9bef-bed132df511d)


```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response = get_completion(prompt)
print(response)
```


---

# GIT
 
## Git Verileri Nasıl Yönetir?

![n2](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/8fb6e3a1-1e1f-4e80-8012-33deced35575)

---

# ChatGPT Prompt Engineering for Developers 

![t1](https://github.com/iremssezer/iremSezerNotebook/assets/74788732/9d96ea8c-5633-4041-94ac-af7d86325178)

```python
Delimiters can be anything like: ```, """, < >, <tag> </tag>
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response = get_completion(prompt)
print(response)
```

---
