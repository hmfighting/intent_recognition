package com.pinyinEnglishClassfy;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

class DataUtil {
    private ArrayList<ArrayList<Double>> alllist = new ArrayList<ArrayList<Double>>(); // Store all data
    private ArrayList<String> outlist = new ArrayList<String>(); // 存放输出数据，索引对应每个everylist的输出
    private ArrayList<String> checklist = new ArrayList<String>();  //存放测试集的真实输出字符串
    private int in_num = 0;
    private int out_num = 0; // 输入输出数据的个数
    private int type_num = 0; // 输出的类型数量
    private double[][] nom_data; //归一化输入数据中的最大值和最小值
    private int in_data_num = 0; //提前获得输入数据的个数

    // Get the number of category
    public int GetTypeNum() {
        return type_num;
    }

    // set the number of category
    public void SetTypeNum(int type_num) {
        this.type_num = type_num;
    }

    // Get the number of input data
    public int GetInNum() {
        return in_num;
    }

    // set the number of input data
    public int GetOutNum() {
        return out_num;
    }

    
    public ArrayList<ArrayList<Double>> GetList() {
        return alllist;
    }

    // 获取输出为字符串形式的数据
    public ArrayList<String> GetOutList() {
        return outlist;
    }

    // 获取输出为字符串形式的数据
    public ArrayList<String> GetCheckList() {
        return checklist;
    }

    //返回归一化数据所需最大最小值
    public double[][] GetMaxMin(){

        return nom_data;
    }

    // 读取文件初始化数据
    public void ReadFile(String filepath, String sep, int flag)
            throws Exception {
        ArrayList<Double> everylist = new ArrayList<Double>(); // 存放每一组输入输出数据
        int readflag = flag; // flag=0,train;flag=1,test
        String encoding = "utf-8";
        File file = new File(filepath);
        if (file.isFile() && file.exists()) { // 判断文件是否存在
            InputStreamReader read = new InputStreamReader(new FileInputStream(
                    file), encoding);// 考虑到编码格式
            BufferedReader bufferedReader = new BufferedReader(read);
            String lineTxt = null;
            //System.out.println("null:"+bufferedReader.readLine());
            while ((lineTxt = bufferedReader.readLine()) != null) {
                int in_number = 0;
                String splits[] = lineTxt.split(sep); // 按','截取字符串
                //System.out.println("lineTxt:"+lineTxt);
                if (readflag == 0) {
                    for (int i = 0; i < splits.length; i++)
                        try {
                        	//System.out.println("i"+i+"max:"+nom_data[i][0]+"min:"+nom_data[i][1]);
                            everylist.add(Normalize(Double.valueOf(splits[i]),nom_data[i][0],nom_data[i][1]));
                            //System.out.println("everylist"+everylist.get(i));
                            in_number++;
                        } catch (Exception e) {
                            if (!outlist.contains(splits[i]))
                                outlist.add(splits[i]); // 存放字符串形式的输出数据
                            for (int k = 0; k < type_num; k++) {
                                everylist.add(0.0);
                            }
                            everylist
                            .set(in_number + outlist.indexOf(splits[i]),
                                    1.0);
                            //System.out.println("everylist"+everylist.get(i));
                            WriteMax(nom_data);
                        }
                } else if (readflag == 1) {
                    for (int i = 0; i < splits.length; i++)
                        try {
                            everylist.add(Normalize(Double.valueOf(splits[i]),nom_data[i][0],nom_data[i][1]));
                            in_number++;
                        } catch (Exception e) {
                            checklist.add(splits[i]); // 存放字符串形式的输出数据
                        }
                }
                alllist.add(everylist); // 存放所有数据
                in_num = in_number;
                out_num = type_num;
                everylist = new ArrayList<Double>();
                everylist.clear();

            }
            bufferedReader.close();
        }
    }
    
    
    public  ArrayList<Double> readVector(String lineTxt){
    	ArrayList<Double> everylist = new ArrayList<Double>(); // 存放每一组输入输出数据
    	singleNormalizeData(lineTxt);
    	String splits[] = lineTxt.split(" "); // 按' '截取字符串
    	try {
			readMax(nom_data);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    	for (int i = 0; i < splits.length; i++)
            try {
                everylist.add(Normalize(Double.valueOf(splits[i]),nom_data[i][0],nom_data[i][1]));
            } catch (Exception e) {
                checklist.add(splits[i]); // 存放字符串形式的输出数据
            }
    	System.out.println("singleeverylist:"+everylist);
    	return everylist;
    }

    //向文件写入分类结果
    public void WriteFile(String filepath, ArrayList<ArrayList<Double>> list, int in_number,  ArrayList<String> resultlist) throws IOException{
        File file = new File(filepath);
        FileWriter fw = null;
        BufferedWriter writer = null;
        try {
            fw = new FileWriter(file);
            writer = new BufferedWriter(fw);
            for(int i=0;i<list.size();i++){
                for(int j=0;j<in_number;j++)
                    writer.write(list.get(i).get(j)+",");
               // System.out.println("resultlist:"+resultlist.size());
                //System.out.println("resultlist.get:"+resultlist.get(i));
                writer.write(resultlist.get(i));
                writer.newLine();
            }
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            writer.close();
            fw.close();
        }
    }


    //学习样本归一化,找到输入样本数据的最大值和最小值
    public void NormalizeData(String filepath) throws IOException{
        //提前获得输入数据的个数   
        GetBeforIn(filepath);
        int flag=1;
        nom_data = new double[in_data_num][2];
        //System.out.println("in_data_num"+in_data_num);
        String encoding = "utf-8";
        File file = new File(filepath);
        if (file.isFile() && file.exists()) { // 判断文件是否存在
            InputStreamReader read = new InputStreamReader(new FileInputStream(
                    file), encoding);// 考虑到编码格式
            BufferedReader bufferedReader = new BufferedReader(read);
            String lineTxt = null;
            int h = 0;
            while ((lineTxt = bufferedReader.readLine()) != null) {
                //String splits[] = lineTxt.split(","); // 按','截取字符串
            	//System.out.println("lineTxt1 "+lineTxt);
            	h++;
            	String splits[] = lineTxt.split(" "); // 按' '截取字符串
                for (int i = 0; i < splits.length-1; i++){
                    if(flag==1){
                        nom_data[i][0]=Double.valueOf(splits[i]);
                        nom_data[i][1]=Double.valueOf(splits[i]);
                        //System.out.println(nom_data[i][0]+" "+nom_data[i][1]);
                    }
                    else{
                        if(Double.valueOf(splits[i])>nom_data[i][0])
                            nom_data[i][0]=Double.valueOf(splits[i]);
                        if(Double.valueOf(splits[i])<nom_data[i][1])
                            nom_data[i][1]=Double.valueOf(splits[i]);
                    }
                }
                flag=0;
            }
//            System.out.println("num:"+h);
//            for(int j = 0; j < 81; j++) {
//            	System.out.println(j+":"+"max:"+nom_data[j][0]+"min:"+nom_data[j][0]);
//            }
           // WriteMax(nom_data);
            bufferedReader.close();
        }
    }
    
    
    public void WriteMax(double[][] nom_data) throws IOException{
    	String filepath = "/Users/fighting/eclipse-workspace/BPNN/data/nom_data.txt";
        File file = new File(filepath);
        if(file.exists()) {
        	file.delete();
        }
        FileWriter fw = null;
        BufferedWriter writer = null;
        try {
            fw = new FileWriter(file);
            writer = new BufferedWriter(fw);
            for(int i = 0; i < nom_data.length; i++) {
                for(int j=0;j<nom_data[i].length;j++)
                    writer.write(nom_data[i][j]+" ");
               // System.out.println("resultlist:"+resultlist.size());
                //System.out.println("resultlist.get:"+resultlist.get(i));
                //writer.write(resultlist.get(i));
                writer.newLine();
            }
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            writer.close();
            fw.close();
        }
    }
    
    
    
 // 读取文件初始化数据
    public void readMax(double[][] nom_data)
            throws Exception {
    	String filepath = "/Users/fighting/eclipse-workspace/BPNN/data/nom_data.txt";
        String encoding = "utf-8";
        File file = new File(filepath);
        int i = 0;
        if (file.isFile() && file.exists()) { // 判断文件是否存在
            InputStreamReader read = new InputStreamReader(new FileInputStream(
                    file), encoding);// 考虑到编码格式
            BufferedReader bufferedReader = new BufferedReader(read);
            String lineTxt = null;
            while ((lineTxt = bufferedReader.readLine()) != null) {
                //int in_number = 0;
            	//System.out.println("lineTxt:"+lineTxt);
                String splits[] = lineTxt.split(" "); // 按' '截取字符串
                
                for(int j = 0; j < splits.length; j++) {
                	
                	if(!splits[j].equals(" ") || !splits[j].equals("")) {
                		nom_data[i][j] = Double.parseDouble(splits[j]);
                		//System.out.println("weight[i][j]"+weight[i][j]);
                	}
                }
                i++;
             }
              
            bufferedReader.close();
        }
    }
    
    public void singleNormalizeData(String lineTxt) {
        nom_data = new double[27][2];
        for(int i = 0; i < 27; i++)
        	nom_data[i][1] = 100;
    	String splits[] = lineTxt.split(" "); // 按' '截取字符串
        for (int i = 0; i < splits.length-1; i++){
        	if(Double.valueOf(splits[i])>nom_data[i][0])
              nom_data[i][0]=Double.valueOf(splits[i]);
          if(Double.valueOf(splits[i])<nom_data[i][1])
              nom_data[i][1]=Double.valueOf(splits[i]);
        }
    }
    
    

    //归一化前获得输入数据的个数
    public void GetBeforIn(String filepath) throws IOException{
        String encoding = "utf-8";
        File file = new File(filepath);
        if (file.isFile() && file.exists()) { // 判断文件是否存在
            InputStreamReader read = new InputStreamReader(new FileInputStream(
                    file), encoding);// 考虑到编码格式
            //提前获得输入数据的个数
            BufferedReader beforeReader = new BufferedReader(read);
            String beforetext = beforeReader.readLine();
            //String splits[] = beforetext.split(",");
            String splits[] = beforetext.split(" ");
            in_data_num = splits.length-1;
            beforeReader.close();
        }
    }

    //归一化公式
    public double Normalize(double x, double max, double min){
    	if(max != 0 || min !=0) {
		    double y = 0.1+0.8*(x-min)/(max-min);
		    return y;
    	}else {
    		return 0.1;
    	}
    }
    
    
  //向文件写入分类结果
    public void WriteWeight(double[][] weight,String flag) throws IOException{
    	String filepath = "/Users/fighting/eclipse-workspace/BPNN/data/"+flag+"Weight.txt";
        File file = new File(filepath);
        if(file.exists()) {
        	file.delete();
        }
        FileWriter fw = null;
        BufferedWriter writer = null;
        try {
            fw = new FileWriter(file);
            writer = new BufferedWriter(fw);
            for(int i=0;i<weight.length;i++){
                for(int j=0;j<weight[i].length;j++)
                    writer.write(weight[i][j]+" ");
               // System.out.println("resultlist:"+resultlist.size());
                //System.out.println("resultlist.get:"+resultlist.get(i));
                //writer.write(resultlist.get(i));
                writer.newLine();
            }
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            writer.close();
            fw.close();
        }
    }
    
    
    
    //向文件写入阀值
    public void WriteTh(double[] th, String flag) throws IOException{
    	String filepath = "/Users/fighting/eclipse-workspace/BPNN/data/"+flag;
        File file = new File(filepath);
        if(file.exists()) {
        	file.delete();
        }
        FileWriter fw = null;
        BufferedWriter writer = null;
        try {
            fw = new FileWriter(file);
            writer = new BufferedWriter(fw);
            
                for(int j=0;j<th.length;j++)
                    writer.write(th[j]+" ");
               // System.out.println("resultlist:"+resultlist.size());
                //System.out.println("resultlist.get:"+resultlist.get(i));
                //writer.write(resultlist.get(i));
                writer.newLine();
            
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            writer.close();
            fw.close();
        }
    }
    
    
    
 
 // 读取文件初始化数据
    public void readWeight(String flag, double[][] weight)
            throws Exception {
    	String filepath = "/Users/fighting/Documents/eclipseCode/BPNN/data/"+flag+"Weight.txt";
        String encoding = "utf-8";
        File file = new File(filepath);
        int i = 0;
        if (file.isFile() && file.exists()) { // 判断文件是否存在
            InputStreamReader read = new InputStreamReader(new FileInputStream(
                    file), encoding);// 考虑到编码格式
            BufferedReader bufferedReader = new BufferedReader(read);
            String lineTxt = null;
            while ((lineTxt = bufferedReader.readLine()) != null) {
                //int in_number = 0;
            	//System.out.println("lineTxt:"+lineTxt);
                String splits[] = lineTxt.split(" "); // 按' '截取字符串
                
                for(int j = 0; j < splits.length; j++) {
                	
                	if(!splits[j].equals(" ") || !splits[j].equals("")) {
                		weight[i][j] = Double.parseDouble(splits[j]);
                		//System.out.println("weight[i][j]"+weight[i][j]);
                	}
                }
                i++;
             }
              
            bufferedReader.close();
        }
    }
    
    
 // 读取文件初始化数据
    public void readTh(String flag, double[] th)
            throws Exception {
    	String filepath = "/Users/fighting/eclipse-workspace/BPNN/data/"+flag;
        String encoding = "utf-8";
        File file = new File(filepath);
        if (file.isFile() && file.exists()) { // 判断文件是否存在
            InputStreamReader read = new InputStreamReader(new FileInputStream(
                    file), encoding);// 考虑到编码格式
            BufferedReader bufferedReader = new BufferedReader(read);
            String lineTxt = null;
            while ((lineTxt = bufferedReader.readLine()) != null) {
//                //int in_number = 0;
//            	System.out.println("lineTxt:"+lineTxt);
                String splits[] = lineTxt.split(" "); // 按' '截取字符串           
                for(int j = 0; j < splits.length; j++) {            	
                	if(!splits[j].equals(" ") || !splits[j].equals("")) {
                		//System.out.println("th:"+th[j]);
                		th[j] = Double.parseDouble(splits[j]);
                	}
                }
             }
              
            bufferedReader.close();
        }
    }
}
