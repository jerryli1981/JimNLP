package edu.pengli.nlp.conference.acl2015.method;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.input.SAXBuilder;

import edu.pengli.nlp.conference.acl2015.generation.AbstractiveGenerator;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RougeEvaluationWrapper;

public class Baseline_IJCAJMethod {
	
	public static void main(String[] args) throws Exception {
		
	    //Create a proxy, which we will use to control MATLAB
		String matlabLocation = "/usr/local/MATLAB/R2012a/bin/matlab";
		MatlabProxyFactoryOptions options = new MatlabProxyFactoryOptions.Builder()
        .setProxyTimeout(30000L).setMatlabLocation(matlabLocation)
        .setHidden(true)
        .build();
		
	    MatlabProxyFactory factory = new MatlabProxyFactory(options);
	    MatlabProxy proxy = factory.getProxy();


		SAXBuilder builder = new SAXBuilder();
		String inputCorpusDir = "../data/ACL2015/testData";
		Document doc = builder.build(inputCorpusDir + "/"
				+ "GuidedSumm_topics.xml");
		Element root = doc.getRootElement();
		List<Element> corpusList = root.getChildren();
		ArrayList<String> corpusNameList = new ArrayList<String>();
		String outputSummaryDir = "../data/ACL2015/Output";
		String modelSummaryDir = "../data/ACL2015/ROUGE/models";
		String confFilePath = "../data/ACL2015/ROUGE/conf.xml";

		
		/*
		 * Pattern Generation
		 */
/*		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		pipeLine.addPipe(new CharSequenceCoreNLPAnnotation());
		pipeLine.addPipe(new RelationExtractionbyOpenIE());
		HeadAnnotation headAnnotator = new HeadAnnotation(); 
		FramenetTagger framenetTagger = new FramenetTagger(); 
		WordnetTagger wordnetTagger = new WordnetTagger();*/
			
		String[] metrics = {"ROUGE-1", "ROUGE-2", "ROUGE-SU4"};
		int[] topNs = {10};
		PrintWriter out = FileOperation.getPrintWriter(new File(outputSummaryDir), 
				"experiment_result");
		
		for(int m=0; m<metrics.length; m++){
			String metric = metrics[m];
			for(int j=0; j<topNs.length; j++){
				int topN = topNs[j];
				double averageMetric = 0.0;
				int iterTime = 2;
				for(int k=0; k<iterTime; k++){
					for (int i = 0; i < corpusList.size(); i++) {				
						System.out.println("Corpus id is "+i);
						Element topic = corpusList.get(i);
						List<Element> docSets = topic.getChildren();
						Element docSetA = docSets.get(1);
						String corpusName = docSetA.getAttributeValue("id");
						corpusNameList.add(corpusName);
						AbstractiveGenerator sg = new AbstractiveGenerator();
						sg.ijcaiMethod(inputCorpusDir + "/" + topic.getAttributeValue("id"),
								outputSummaryDir, corpusName, topN, proxy);
					}
					
					// Rouge Evaluation
					ArrayList<File> files = FileOperation.travelFileList(new File(
							modelSummaryDir));
					HashMap<String, ArrayList<String>> modelSummariesMap = new HashMap<String, ArrayList<String>>();
					ArrayList<String> list = null;
					for (File f : files) {
						String fn = f.getName();
						String[] toks = fn.split("\\.");
						String idx = toks[0].split("-")[0];
						String abb = idx + toks[toks.length - 2] + "-"
								+ toks[0].split("-")[1];
						if (corpusNameList.contains(abb)) {

							if (!modelSummariesMap.containsKey(abb)) {
								list = new ArrayList<String>();
								list.add(fn);
							} else {
								list = modelSummariesMap.get(abb);
								list.add(fn);
							}
							modelSummariesMap.put(abb, list);

						}
					}
					
					RougeEvaluationWrapper.setConfigurationFile(corpusNameList,
							outputSummaryDir, modelSummaryDir, modelSummariesMap,
							confFilePath);
					
					HashMap map = RougeEvaluationWrapper.runRough(confFilePath, metric);
					Double met = (Double) map.get(metric);
					System.out.println(metric+" "+topN+" "+met);
					averageMetric += met;
				}
				
				System.out.println(metric + " "+ topN + " : " + averageMetric/iterTime);
				out.println(metric + " "+ topN + " : " + averageMetric/iterTime);
			}

		}
		
		proxy.disconnect();
		out.close();
	}

}
