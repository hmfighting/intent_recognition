package com.pinyinEnglishClassfy;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

class BPNN {
	private static int NodeNum = 10; // maximum number of nodes for each layer
	private static final int ADJUST = 5; // adjustment constant of the number of hidden nodes
	private static final int MaxTrain = 2000; // Maximum training times
	private static final double ACCU = 0.015; // Allowable error for per iteration :0.015
	private double ETA_W = 0.5; // the efficiency of weight learning: 0.5
	private double ETA_T = 0.5; // efficiency of threshold learning
	private double accu;


	private int in_num; // Number of input layer nodes
	private int hd_num; // Number of hidden layer nodes
	private int out_num; // Number of output layer nodes

	private ArrayList<ArrayList<Double>> list = new ArrayList<>(); // 输入输出数据
	private DataUtil datautil = new DataUtil();
	private double[][] in_hd_weight; // weight value between input layer and hidden layer
	private double[][] hd_out_weight; // weight value between hidden layer and output layer
	private double[] in_hd_th; // threshold between in-hidden
	private double[] hd_out_th; // threshold between hidden-out

	private double[][] out; // The output value of each neuron transformed by the s-shaped function. Where, it is original value in input layer
	private double[][] delta; // delta学习规则中的值

	//obtain the maximum number of neurons in the three layers of the network
	public int GetMaxNum() {
		return Math.max(Math.max(in_num, hd_num), out_num);
	}

	// Set the weight learning rate
	public void SetEtaW() {
		ETA_W = 0.5;
	}

	// Set the threshold learning rate
	public void SetEtaT() {
		ETA_T = 0.5;
	}

	// BPNN training
	public void Train(int in_number, int out_number, ArrayList<ArrayList<Double>> arraylist) throws IOException {

		list = arraylist;
		in_num = in_number;
		out_num = out_number;

		GetNums(in_num, out_num); // get the nodes of input layer, hidden layer and output layer

		InitNetWork(); // initial the weight values and threshold values初始化网络的权值和阈值

		int datanum = list.size(); // the number of training data
		int createsize = GetMaxNum(); // 比较创建存储每一层输出数据的数组
		out = new double[3][createsize];

		for (int iter = 0; iter < MaxTrain; iter++) {
			for (int cnd = 0; cnd < datanum; cnd++) {
				// 第一层输入节点赋值

				for (int i = 0; i < in_num; i++) {
					out[0][i] = list.get(cnd).get(i); // 为输入层节点赋值，其输入与输出相同
				}
				Forward(); 
				Backward(cnd); 

			}
			System.out.println("This is the " + (iter + 1) + " th trainning NetWork !");
			accu = GetAccu();
			System.out.println("All Samples Accuracy is " + accu);
			if (accu < ACCU)
				break;

		}
		datautil.WriteWeight(in_hd_weight, "in");
		datautil.WriteWeight(hd_out_weight, "hd");
		datautil.WriteTh(in_hd_th, "in_hd_th.txt");
		datautil.WriteTh(hd_out_th, "hd_out_th.txt");
		System.out.println("train" + in_num + ":" + out_num);
	}

	// 获取输入层、隐层、输出层的节点数，in_number、out_number分别为输入层节点数和输出层节点数
	public void GetNums(int in_number, int out_number) {
		in_num = in_number;
		out_num = out_number;
		hd_num = (int) Math.sqrt(in_num + out_num) + ADJUST;
		if (hd_num > NodeNum)
			hd_num = NodeNum; // The number of hidden nodes cannot be greater than the maximum number of nodes
	}

	// Initializes weights and thresholds for the network
	public void InitNetWork() {
		in_hd_weight = new double[in_num][hd_num];
		for (int i = 0; i < in_num; i++)
			for (int j = 0; j < hd_num; j++) {
				int flag = 1; // 符号标志位(-1或者1)
				if ((new Random().nextInt(2)) == 1)
					flag = 1;
				else
					flag = -1;
				in_hd_weight[i][j] = (new Random().nextDouble() / 2) * flag; // 初始化in-hidden的权值
				// in_hd_last[i][j] = 0;
			}

		hd_out_weight = new double[hd_num][out_num];
		for (int i = 0; i < hd_num; i++)
			for (int j = 0; j < out_num; j++) {
				int flag = 1; // 符号标志位(-1或者1)
				if ((new Random().nextInt(2)) == 1)
					flag = 1;
				else
					flag = -1;
				hd_out_weight[i][j] = (new Random().nextDouble() / 2) * flag; // 初始化hidden-out的权值
				// hd_out_last[i][j] = 0;
			}

		// thresholds are initialized to 0
		in_hd_th = new double[hd_num];
		for (int k = 0; k < hd_num; k++)
			in_hd_th[k] = 0;

		hd_out_th = new double[out_num];
		for (int k = 0; k < out_num; k++)
			hd_out_th[k] = 0;

	}

	// Calculate the error of a single sample
	public double GetError(int cnd) {
		double ans = 0;
		for (int i = 0; i < out_num; i++)
			ans += 0.5 * (out[2][i] - list.get(cnd).get(in_num + i)) * (out[2][i] - list.get(cnd).get(in_num + i));
		return ans;
	}

	// Calculate the average accuracy of all samples
	public double GetAccu() {
		double ans = 0;
		int num = list.size();
		for (int i = 0; i < num; i++) {
			int m = in_num;
			for (int j = 0; j < m; j++)
				out[0][j] = list.get(i).get(j);
			Forward();
			int n = out_num;
			for (int k = 0; k < n; k++)
				ans += 0.5 * (list.get(i).get(in_num + k) - out[2][k]) * (list.get(i).get(in_num + k) - out[2][k]);
		}
		return ans / num;
	}

	// The forward propagation
	public void Forward() {
		// Calculates the output value of the hidden layer node
		for (int j = 0; j < hd_num; j++) {
			double v = 0;
			for (int i = 0; i < in_num; i++) {
				// System.out.println("in_hd_weight:"+in_hd_weight[i][j]);
				v += in_hd_weight[i][j] * out[0][i];
			}
			v += in_hd_th[j];
			out[1][j] = Sigmoid(v);
		}
		// Calculates the output value of the output layer node
		for (int j = 0; j < out_num; j++) {
			double v = 0;
			for (int i = 0; i < hd_num; i++)
				v += hd_out_weight[i][j] * out[1][i];
			v += hd_out_th[j];
			out[2][j] = Sigmoid(v);
		}
	}

	//Backword Propagation
	public void Backward(int cnd) {
		CalcDelta(cnd); // Calculate the weight adjustment value
		UpdateNetWork(); // Update weights and thresholds of BP neural network
	}

	// Calculate the delta adjustment value
	public void CalcDelta(int cnd) {

		int createsize = GetMaxNum(); 
		delta = new double[3][createsize];
		// Calculate the delta value of the output layer
		for (int i = 0; i < out_num; i++) {
			delta[2][i] = (list.get(cnd).get(in_num + i) - out[2][i]) * SigmoidDerivative(out[2][i]);
		}

		// Calculate the delta value of the hidden layer
		for (int i = 0; i < hd_num; i++) {
			double t = 0;
			for (int j = 0; j < out_num; j++)
				t += hd_out_weight[i][j] * delta[2][j];
			delta[1][i] = t * SigmoidDerivative(out[1][i]);
		}
	}

	// Update weights and thresholds of BP neural network
	public void UpdateNetWork() {

		// Weight and threshold adjustments between the hidden layer and the output layer
		for (int i = 0; i < hd_num; i++) {
			for (int j = 0; j < out_num; j++) {
				hd_out_weight[i][j] += ETA_W * delta[2][j] * out[1][i]; 
			}

		}
		for (int i = 0; i < out_num; i++)
			hd_out_th[i] += ETA_T * delta[2][i];

		// Weight and threshold adjustments between the input layer and the hidden layer
		for (int i = 0; i < in_num; i++) {
			for (int j = 0; j < hd_num; j++) {
				in_hd_weight[i][j] += ETA_W * delta[1][j] * out[0][i];
			}
		}
		for (int i = 0; i < hd_num; i++)
			in_hd_th[i] += ETA_T * delta[1][i];
	}

	// sign function
	public int Sign(double x) {
		if (x > 0)
			return 1;
		else if (x < 0)
			return -1;
		else
			return 0;
	}

	// return maximum
	public double Maximum(double x, double y) {
		if (x >= y)
			return x;
		else
			return y;
	}

	// return minimum
	public double Minimum(double x, double y) {
		if (x <= y)
			return x;
		else
			return y;
	}

	// log-sigmoid function
	public double Sigmoid(double x) {
		return (double) (1 / (1 + Math.exp(-x)));
	}

	// the inverse of log-sigmoid function
	public double SigmoidDerivative(double y) {
		return (double) (y * (1 - y));
	}

	// tan-sigmoid function
	public double TSigmoid(double x) {
		return (double) ((1 - Math.exp(-x)) / (1 + Math.exp(-x)));
	}

	// the inverse of tan-sigmoid function
	public double TSigmoidDerivative(double y) {
		return (double) (1 - (y * y));
	}

	// Classification prediction function
	public ArrayList<ArrayList<Double>> ForeCast(ArrayList<ArrayList<Double>> arraylist) {
//System.out.println("in_num:"+in_num);
//System.out.println("out_num:"+out_num);
		// 预测初始化
		in_num = 26;
		out_num = 2;
		GetNums(in_num, out_num);
		int createsize = GetMaxNum(); // 比较创建存储每一层输出数据的数组
		out = new double[3][createsize];
		in_hd_weight = new double[in_num][hd_num];
		hd_out_weight = new double[hd_num][out_num];
		in_hd_th = new double[hd_num];
		hd_out_th = new double[out_num];

		try {
			datautil.readTh("in_hd_th.txt", in_hd_th);
			datautil.readTh("hd_out_th.txt", hd_out_th);
			datautil.readWeight("in", in_hd_weight);
			datautil.readWeight("hd", hd_out_weight);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		ArrayList<ArrayList<Double>> alloutlist = new ArrayList<>();
		ArrayList<Double> outlist = new ArrayList<Double>();
		int datanum = arraylist.size();
		for (int cnd = 0; cnd < datanum; cnd++) {
			for (int i = 0; i < in_num; i++) {
				// System.out.println("outsize:"+out[0].length);
				out[0][i] = arraylist.get(cnd).get(i); // 为输入节点赋值
			}
			Forward();
			for (int i = 0; i < out_num; i++) {
				if (out[2][i] > 0 && out[2][i] < 0.5)
					out[2][i] = 0;
				else if (out[2][i] > 0.5 && out[2][i] < 1) {
					out[2][i] = 1;
				}
				outlist.add(out[2][i]);
			}
			alloutlist.add(outlist);
			outlist = new ArrayList<Double>();
			outlist.clear();
		}
		return alloutlist;
	}

	// 单句预测函数
	public int singleForeCast(ArrayList<Double> arraylist) {
		// initialization
		in_num = 26;
		out_num = 2;
		GetNums(in_num, out_num);
		int createsize = GetMaxNum(); // 比较创建存储每一层输出数据的数组
		out = new double[3][createsize];
		in_hd_weight = new double[in_num][hd_num];
		hd_out_weight = new double[hd_num][out_num];
		in_hd_th = new double[hd_num];
		hd_out_th = new double[out_num];

		try {
			datautil.readTh("in_hd_th.txt", in_hd_th);
			datautil.readTh("hd_out_th.txt", hd_out_th);
			datautil.readWeight("in", in_hd_weight);
			datautil.readWeight("hd", hd_out_weight);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int result = -1;
		ArrayList<Double> outlist = new ArrayList<Double>();
		// for (int cnd = 0; cnd < datanum; cnd++) {
		
		for (int i = 0; i < in_num; i++)
			out[0][i] = arraylist.get(i); // 为输入节点赋值
		Forward();
		for (int i = 0; i < out_num; i++) {
			if (out[2][i] > 0 && out[2][i] < 0.5)
				out[2][i] = 0;
			else if (out[2][i] > 0.5 && out[2][i] < 1) {
				out[2][i] = 1;
			}
			outlist.add(out[2][i]);
		}
		for(int i = 0; i < outlist.size(); i++) {
			if(outlist.get(i) == 1) {
				if(i == 0) {
					result = 1;  //English
				}
				if(i == 1) {
					result = 0; //pinyin
				}
			}
		}
		return result;
	}

}