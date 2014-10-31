package edu.pengli.nlp.conference.acl2015.pipe;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;

import org.json.JSONArray;
import org.json.JSONObject;

import scala.collection.Iterator;
import scala.collection.immutable.SortedSet;
import edu.cmu.cs.lti.ark.fn.Semafor;
import edu.knowitall.tool.parse.MaltParser;
import edu.knowitall.tool.parse.graph.Dependency;
import edu.knowitall.tool.parse.graph.DependencyGraph;
import edu.knowitall.tool.parse.graph.DependencyNode;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.postag.Postagger;
import edu.knowitall.tool.tokenize.ClearTokenizer;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.stanford.nlp.ling.CoreLabel;


public class FramenetTagger {
	
	MaltParser maltParser;
	
	Semafor tagger;
	
	public FramenetTagger(){
		
		String MALT_PARSER_FILENAME = "/home/peng/Develop/Workspace/"
				+ "Mavericks/models/malparser/engmalt.linear-1.7.mco";
		Postagger nlppostagger = new ClearPostagger(new ClearTokenizer(ClearTokenizer.defaultModelUrl()));
		scala.Option<File> nullOption = scala.Option.apply(null);
		try {
			maltParser = new MaltParser(new File(MALT_PARSER_FILENAME).toURI()
					.toURL(), nlppostagger, nullOption);
		} catch (MalformedURLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String modelDirectory = "/home/peng/Develop/Workspace/Mavericks/models/SEMAFOR/models";
		try {
			tagger = Semafor.getSemaforInstance(modelDirectory);
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (URISyntaxException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private String generateSemaforInput(String sentence){
				
			DependencyGraph graph = maltParser.apply(sentence);

			StringBuilder sb = new StringBuilder();
			SortedSet<DependencyNode> set = graph.nodes();
			SortedSet<Dependency> dependencies = graph.dependencies();
			
			Iterator it = set.iterator();
			while (it.hasNext()) {
				DependencyNode n = (DependencyNode) it.next();
				String mention = n.string();
				String postag = n.postag();
				int idx = n.index() + 1;
				Iterator it_dep = dependencies.iterator();
				DependencyNode source = null;
				String relation = null;
				while (it_dep.hasNext()) {
					Dependency dependency = (Dependency) it_dep.next();
					DependencyNode dest = dependency.dest();
					if (n.equals(dest)) {
						source = dependency.source();
						relation = dependency.label();
						break;
					}
				}

				if (source != null) {

					sb.append(idx + "\t" + mention + "\t_\t" + postag + "\t"
							+ postag + "\t_\t" + (source.index() + 1) + "\t"
							+ relation + "\t_\t_");

				} else {
					sb.append(idx + "\t" + mention + "\t_\t" + postag + "\t"
							+ postag + "\t_\t" + 0 + "\t" + null + "\t_\t_");
				}

				sb.append("\n");
			}
			
			return sb.toString().trim();
			
	}
	
	
	public void annotate(CoreLabel arg1Head, CoreLabel arg2Head, Tuple t) throws Exception{
		
		String tupleMention = arg1Head.originalText()+" "+t.getRel().toString()+" "+arg2Head.originalText();
		
		String conllFormat = generateSemaforInput(tupleMention);
		
		String json = tagger.annotateSentence(conllFormat);
		
		JSONObject jsonObj = new JSONObject(json);
		JSONArray frames = (JSONArray) jsonObj.get("frames");
		for (int i = 0; i < frames.length(); i++) {
			JSONObject frame = frames.getJSONObject(i);
			JSONArray annotationSet = (JSONArray)frame.get("annotationSets");
//			JSONObject target = frame.getJSONObject("target");
//			String frameName = target.getString("name");
			for(int j=0; j<annotationSet.length(); j++){
				JSONObject annotation = annotationSet.getJSONObject(j);
				JSONArray frameelements = (JSONArray)annotation.get("frameElements");
				for(int k=0; k<frameelements.length(); k++){
					JSONObject felement = frameelements.getJSONObject(k);
					JSONArray spans = (JSONArray)felement.get("spans");
					String labelName = felement.getString("name");
					for(int l=0; l<spans.length(); l++){
						JSONObject span = spans.getJSONObject(l);
						String text = span.getString("text");
						if(text.equals(arg1Head.originalText()) && arg1Head.ner().equals("O")){
							arg1Head.setNER(labelName);
						}
						
						if(text.equals(arg2Head.originalText()) && arg2Head.ner().equals("O")){
							arg2Head.setNER(labelName);
						}
					}
				}	
			}
		}
	
	}

	public static void main(String[] args) throws Exception{

	
	}
}