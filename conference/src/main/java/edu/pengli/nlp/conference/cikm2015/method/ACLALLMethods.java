package edu.pengli.nlp.conference.cikm2015.method;

import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;

import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.input.SAXBuilder;

import edu.pengli.nlp.conference.cikm2015.generation.AbstractiveGenerator;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;

public class ACLALLMethods {
	

	public static void main(String[] args) throws Exception {

		// Create a proxy, which we will use to control MATLAB
		String matlabLocation = "/usr/local/MATLAB/R2012a/bin/matlab";
		MatlabProxyFactoryOptions options = new MatlabProxyFactoryOptions.Builder()
				.setProxyTimeout(30000L).setMatlabLocation(matlabLocation)
				.setHidden(true).build();

		MatlabProxyFactory factory = new MatlabProxyFactory(options);
		MatlabProxy proxy = factory.getProxy();

		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();

		String outputSummaryDir = "../data/ACL2015/Output";
		String confFilePath = "../data/ACL2015/ROUGE/conf.xml";
		

		int iterTime = 3;
				
		///////////////////////////////////////////////	
		
		int[] numberClusters_DCNN = {7, 8};
		

		//////////////////////////////////////////////////////
		PrintWriter out_DCNN_Spe_H_Har_H_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_H_Har_H_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_H_Har_H_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_H_Har_H_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_H_Har_H_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_H_Har_H_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_H_Har_H_110.close();
		
		/////////////////////////////////////////////
		
		PrintWriter out_DCNN_Spe_H_Har_C_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_H_Har_C_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_H_Har_C_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_H_Har_C_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_H_Har_C_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_H_Har_C_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_H_Har_C_110.close();
		
		////////////////////////////////////
		PrintWriter out_DCNN_Spe_C_Gre_H_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_C_Gre_H_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_C_Gre_H_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_C_Gre_H_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_C_Gre_H_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_C_Gre_H_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_C_Gre_H_110.close();
		
		//////////////////////////////
		PrintWriter out_DCNN_Spe_C_Gre_C_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_C_Gre_C_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_C_Gre_C_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_C_Gre_C_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_C_Gre_C_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_C_Gre_C_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_C_Gre_C_110.close();
		////////////////////////////////
		PrintWriter out_DCNN_Spe_H_Gre_H_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_H_Gre_H_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_H_Gre_H_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_H_Gre_H_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_H_Gre_H_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_H_Gre_H_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_H_Gre_H_110.close();
		
		////////////////////////
		PrintWriter out_DCNN_Spe_H_Gre_C_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_H_Gre_C_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_H_Gre_C_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_H_Gre_C_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_H_Gre_C_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_H_Gre_C_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_H_Gre_C_110.close();
		
		////////////////////////////
		PrintWriter out_DCNN_Spe_H_LGC_C_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_H_LGC_C_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_H_LGC_C_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_H_LGC_C_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_H_LGC_C_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_H_LGC_C_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_H_LGC_C_110.close();
		
		///////////////////////
		
		PrintWriter out_DCNN_Spe_C_LGC_C_110 = FileOperation.getPrintWriter(new File(
				outputSummaryDir), "experiment_result_DCNN_Spe_C_LGC_C_110");
		for (int j = 0; j < numberClusters_DCNN.length; j++) {
			int numberCluster = numberClusters_DCNN[j];
			double averageMetric_1 = 0.0;
			double averageMetric_2 = 0.0;
			double averageMetric_SU4 = 0.0;

			for (int k = 0; k < iterTime; k++) {
				for (int i = 0; i < corpusList.size(); i++) {
					System.out.println("Corpus id is " + i);
					Element topic = corpusList.get(i);
					List<Element> docSets = topic.getChildren();
					Element docSetA = docSets.get(1);
					String corpusName = docSetA.getAttributeValue("id");
					AbstractiveGenerator sg = new AbstractiveGenerator();
					sg.DCNN_Spe_C_LGC_C_110(inputCorpusDir + "/" +topic.getAttributeValue("id"), outputSummaryDir, corpusName, numberCluster, proxy);
					 
				}

				HashMap map_1 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-1");
				Double met_1 = (Double) map_1.get("ROUGE-1");
				System.out.println("ROUGE-1" + " " + numberCluster + " " + met_1);
				averageMetric_1 += met_1;

				HashMap map_2 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-2");
				Double met_2 = (Double) map_2.get("ROUGE-2");
				System.out.println("ROUGE-2" + " " + numberCluster + " " + met_2);
				averageMetric_2 += met_2;

				HashMap map_SU4 = RougeEvaluationWrapper.runRough(confFilePath,
						"ROUGE-SU4");
				Double met_SU4 = (Double) map_SU4.get("ROUGE-SU4");
				System.out.println("ROUGE-SU4" + " " + numberCluster + " " + met_SU4);
				averageMetric_SU4 += met_SU4;

			}

			System.out.println("Average ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);
			out_DCNN_Spe_C_LGC_C_110.println("Average  ROUGE-1" + " " + numberCluster + " : " + averageMetric_1
					/ iterTime);

			System.out.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);
			out_DCNN_Spe_C_LGC_C_110.println("Average ROUGE-2" + " " + numberCluster + " : " + averageMetric_2
					/ iterTime);

			System.out.println("Average ROUGE-SU4" + " " + numberCluster + " : "
					+ averageMetric_SU4 / iterTime);
			out_DCNN_Spe_C_LGC_C_110.println("Average ROUGE-SU4" + " " + numberCluster + " : " + averageMetric_SU4
					/ iterTime);
		}
		out_DCNN_Spe_C_LGC_C_110.close();
		
		
		proxy.disconnect();
	}

}
