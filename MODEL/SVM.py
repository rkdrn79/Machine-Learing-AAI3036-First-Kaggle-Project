__author__ = "Mingu Kang"
__id__ = "2022313618"

# Do not import and other Python libraries
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Write your code following the instructions
class SVMClassifier:
    def __init__(self,n_iters=100, lr = 0.0001, random_seed=3, lambda_param=0.01):
        """
        This function doesn't need to be modified.
        """
        self.author = __author__
        self.id = __id__
        self.n_iters = n_iters # number of iterations
        self.lr = lr  # learning rate
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


    def fit(self, x, y):
        """
        This function trains the model using x, y.
        You can reference the website below for gradient updates.
        reference: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
        Fill out the 6 "None"s from the code below.

        """
        n_samples, n_features = x.shape

        # hint: in order to use y for SVM, change zeros to -1.
        y_ = np.where(y <= 0, -1, 1)
        
        # hint: reset w, a numpy array with random values between 0 to 1, with the size of (n_features, ).
        init_w = np.random.rand(n_features) 
        self.w = init_w
        self.b = 0 # reset b

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = x[i]
                y_i = y_[i]

                # hint: filter cases with y(i) * (w · x(i) + b) >= 1 using if statement.
                condition = y_i * (np.dot(self.w, x_i) + self.b) >= 1 
                if condition:
                    # hint: update W using the Gradient Loss Function equation.
                    self.w -= self.lr * (2 * self.lambda_param * self.w) 
                else:
                    # hint: update W using the Gradient Loss Function equation.
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_i)) 
                    self.b -= self.lr * y_i

        return self.w, self.b

    def predict(self, x):
        """
            Given x, [n_samples x features], use self.w and self.b from fit() to predict the value.

            @args:
                x with the shape of [n_samples x features]
            @returns:
                array with the shape of [n_samples, ]

            You can refer to the equation and pseudocode below:
                approximation = W·X - b <= 해당 주석에 오류가 있음. fit을 하는 과정에서 bias를 더해줬지만, 해당 주석대로 실행한다면 predict를 할때 bias를 빼야함. 따라서 W·X + b 로 수정해야함
                if approximation >= 0 {
                    output = 1
                }
                else{
                    output = 0
                }
        """
        approximations = np.dot(x, self.w) + self.b
        return approximations

    def get_accuracy(self, y_true, y_pred):
        """
            Calcuate the accuracy using y_true and y_pred.
            Do not use sklearn's accuracy_score. Only use numpy.
        """
        return np.mean(y_true == y_pred)

class MulticlassSVMClassifier(SVMClassifier):
    def __init__(self, n_iters=100, lr=0.0001, random_seed=3, lambda_param=0.01):
        super().__init__(n_iters, lr, random_seed, lambda_param)
        self.classes = None
        self.classifiers = {}

    def fit(self, x, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            binary_y = np.where(y == cls, 1, -1)  # 클래스에 해당하는 것은 1, 나머지는 -1로 처리
            self.classifiers[cls] = SVMClassifier(n_iters=self.n_iters, lr=self.lr,
                                                         random_seed=self.random_seed, lambda_param=self.lambda_param)
            self.classifiers[cls].fit(x, binary_y)

    def predict(self, x):
        predictions = []
        for _, classifier in self.classifiers.items():
            prediction = classifier.predict(x)
            predictions.append(prediction)
        final_predictions = np.argmax(np.array(predictions), axis=0)
        return final_predictions
    

    """
    def generate_combinations(self, elements, r):
        combinations = []
        self.generate_combinations_util(elements, r, 0, [], combinations)
        return combinations

    def generate_combinations_util(self, elements, r, index, combination, combinations):
        if len(combination) == r:
            combinations.append(tuple(combination))
            return

        for i in range(index, len(elements)):
            self.generate_combinations_util(elements, r, i + 1, combination + [elements[i]], combinations)
            
    def fit(self, x, y, x_val, y_val):
        self.classes = np.unique(y)

        # 각 클래스를 고정하고 나머지 클래스들과의 모든 조합을 반복하여 훈련 및 검증
        for fixed_class in self.classes:
            best_accuracy = 0
            best_combination = None

            other_classes = np.delete(self.classes, np.where(self.classes == fixed_class))

            # 모든 가능한 조합 생성
            for num_classes in range(1, len(other_classes) + 1):
                combinations = self.generate_combinations(other_classes, num_classes)
                for combination in combinations:
                    indices = np.where(np.isin(y, [fixed_class] + list(combination)))
                    binary_y = np.where(y[indices] == fixed_class, 1, -1)
                    binary_y_val = np.where(y_val == fixed_class, 1, -1)

                    classifier = SVMClassifier(n_iters=self.n_iters, lr=self.lr,
                                               random_seed=self.random_seed, lambda_param=self.lambda_param)
                    classifier.fit(x[indices], binary_y)

                    # 검증 성능 측정
                    predictions = classifier.predict(x_val)
                    accuracy = classifier.get_accuracy(binary_y_val, np.where(predictions >= 0, 1, -1))

                    # 최고의 성능을 보이는 조합 업데이트
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy ㄴ
                        best_combination = combination

            # 최고의 조합으로 전체 분류기 훈련
            indices = np.where(np.isin(y, [fixed_class] + list(best_combination)))
            binary_y = np.where(y[indices] == fixed_class, 1, -1)
            self.classifiers[fixed_class] = SVMClassifier(n_iters=self.n_iters, lr=self.lr,
                                                    random_seed=self.random_seed, lambda_param=self.lambda_param)
            self.classifiers[fixed_class].fit(x[indices], binary_y)

    def predict(self, x):
        all_prediction = []
        for _, classifier in self.classifiers.items():
            prediction = classifier.predict(x)
            all_prediction.append(prediction)

        # Softmax 함수를 사용하여 확률 분포 생성
        probabilities = self.softmax(all_prediction)

        # 각 샘플에 대해 가장 높은 확률을 가진 클래스 선택
        final_predictions = np.argmax(probabilities, axis=0)
        return final_predictions
    """ 
