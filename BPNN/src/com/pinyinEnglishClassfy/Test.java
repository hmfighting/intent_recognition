package com.pinyinEnglishClassfy;

import java.io.IOException;
import java.util.ArrayList;

public class Test {
	public static DataUtil dataUtil = new DataUtil(); // initial data
	public static BPNN bpnn = new BPNN();
	public static void train(BPNN bpnn, int in_num, int out_num, ArrayList<ArrayList<Double>> alllist) {
      System.out.println();
      System.out.println("训练集的数量："+alllist.size());

      //BPNN bpnn = new BPNN();
      // 训练
      System.out.println("Train Start!");
      System.out.println(".............");
      try {
		bpnn.Train(in_num, out_num, alllist);
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
      System.out.println("Train End!");
	}
	
	
	//将字符串和标签转化为向量
		public  static String txtToVector(String line) {
			String str = "";
			int[] inMatrix = new int[27];
			String[] sarr = line.split(" ");
			for (int i = 0; i < sarr.length; i++) {
//				if (i == sarr.length - 1) {
//					inMatrix[26] = Integer.parseInt(sarr[sarr.length - 1]);
//				} else {
					sarr[i] = sarr[i].toLowerCase();
					char[] carr = sarr[i].toCharArray();
					int n = carr[0] - 'a';
					System.out.println(n);
					int sum = 0;
					if(i == 0) {
					for (int j = 1; j < carr.length; j++) {
						if(carr[j]>='a'&&carr[j]<='z') {
						System.out.println("char:"+carr[j]);
						sum = sum + carr[j] - 'a';
						}
					}
					inMatrix[n] = sum;
					}else {
						for (int j = 0; j < carr.length; j++) {
							if(carr[j]>='a'&&carr[j]<='z') {
							System.out.println("char:"+carr[j]);
							sum = sum + carr[j] - 'a';
							}
						}
						inMatrix[n] = sum;
					}
				//}
			}
			int j = 0;
			for (; j < 25; j++) {
				System.out.print(inMatrix[j] + " ");
				str = str + inMatrix[j] + " ";
			}
			if(j == 25) {
				str = str + inMatrix[j];
			}
			//System.out.println();
			return str;
		}
		
		public static int getLineResult(String line) {
			int result = -1;
			String temp = txtToVector(line);
			ArrayList<Double> arraylist = dataUtil.readVector(temp);
			result = bpnn.singleForeCast(arraylist);
			return result;
		}
		
		/**
		 *
		 * @param str
		 * @return 0：拼音  1:英文
		 */
		public static int isPinyinOrEnglist(String str) { 
	        int lineResult = getLineResult(str);
	        return lineResult;
		}
		
    public static void main(String args[]) throws Exception {
    	
    	
        ArrayList<ArrayList<Double>> alllist = new ArrayList<ArrayList<Double>>(); // 存放所有数据
        ArrayList<String> outlist = new ArrayList<String>(); // 存放分类的字符串
        int in_num = 0, out_num = 0; // 输入输出数据的个数

        //DataUtil dataUtil = new DataUtil(); // 初始化数据
        dataUtil.NormalizeData("/Users/fighting/eclipse-workspace/BPNN/data/yuliaoMatrix.txt");
        dataUtil.SetTypeNum(2); // 设置输出类型的数量
        //dataUtil.SetTypeNum(3); // 设置输出类型的数量
        dataUtil.ReadFile("/Users/fighting/eclipse-workspace/BPNN/data/yuliaoMatrix.txt", " ", 0);
        in_num = dataUtil.GetInNum(); // 获得输入数据的个数
        out_num = dataUtil.GetOutNum(); // 获得输出数据的个数(个数代表类型个数)
        alllist = dataUtil.GetList(); // 获得初始化后的数据
        //System.out.println("size:"+alllist.size());
        outlist = dataUtil.GetOutList();
        System.out.print("分类的类型：");
        for(int i =0 ;i<outlist.size();i++)
            System.out.print(outlist.get(i)+"  ");
       
        
        //train the model 
        train(bpnn,in_num, out_num, alllist);


        // test the model
        DataUtil testUtil = new DataUtil();

        testUtil.NormalizeData("/Users/fighting/eclipse-workspace/BPNN/data/yuliaoMatrixTest.txt");
        testUtil.SetTypeNum(2); // set the category number
        testUtil.ReadFile("/Users/fighting/eclipse-workspace/BPNN/data/yuliaoMatrixTest.txt", " ", 1);
        ArrayList<ArrayList<Double>> testList = new ArrayList<ArrayList<Double>>();
        ArrayList<ArrayList<Double>> resultList = new ArrayList<ArrayList<Double>>();
        ArrayList<String> normallist = new ArrayList<String>(); // 存放测试集标准的输出字符串
        ArrayList<String> resultlist = new ArrayList<String>(); // 存放测试集计算后的输出字符串

        double right = 0; // amount of data which is correctly classified
        int type_num = 0; // category number
        double all_num = 0; //the amount of the test data
        type_num = outlist.size();

        testList = testUtil.GetList(); // get the test data
                     
        normallist = testUtil.GetCheckList(); 
        int errorcount = 0; // amount of data which is misclassified
        resultList = bpnn.ForeCast(testList); //prediction
       // System.out.println("resultListsize:"+resultList.size());
        all_num=resultList.size();
        for (int i = 0; i < resultList.size(); i++) {
            String checkString = "unknow";
            for (int j = 0; j < type_num; j++) {
                if(resultList.get(i).get(j)==1.0){
                    checkString = outlist.get(j);
                    resultlist.add(checkString);
                }
            }
            if(checkString.equals(normallist.get(i)))
                right++;
        }
        testUtil.WriteFile("/Users/fighting/eclipse-workspace/BPNN/data/result.txt",testList,in_num,resultlist);

        System.out.println("the amount of test dataset："+ (new Double(all_num)).intValue());
        System.out.println("amount of data which is correctly classified："+(new Double(right)).intValue());
        System.out.println("accuracy："+right/all_num);
        System.out.println("the place that prediction results are saved：/Users/fighting/eclipse-workspace/BPNN/data/result.txt");      
        int lineResult = getLineResult("wo de jian wei shen me hai mei dao");
        
        //int lineResult = isPinyinOrEnglist("shou zhong shi shen me");
        if(lineResult == 0) {
        	System.out.println("result:pinyin");
        }else {
        	System.out.println("result:English");
        }
        
        
        //0 9 0 0 0 0 10 0 0 0 38 15 0 0 0 0 0 0 0 0 0 0 0 0 32 0 0 
//        arraylist.add(0);
//        arraylist.add(9);
//        arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(10);
//        arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(38);arraylist.add(15);
//        arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(0);
//        arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(0);arraylist.add(0);
//        arraylist.add(0);arraylist.add(0);arraylist.add(32);arraylist.add(0);arraylist.add(0);
//        System.out.println(bpnn.singleForeCast(arraylist));
    }
}