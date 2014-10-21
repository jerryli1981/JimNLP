package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.TreeMap;

import scala.collection.Iterator;
import scala.collection.Seq;
import edu.knowitall.collection.immutable.Interval;
import edu.knowitall.openie.Argument;
import edu.knowitall.openie.OpenIE;
import edu.knowitall.openie.Relation;
import edu.knowitall.tool.parse.ClearParser;
import edu.knowitall.tool.postag.ClearPostagger;
import edu.knowitall.tool.srl.ClearSrl;
import edu.knowitall.tool.tokenize.ClearTokenizer;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.Instance;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class RelationExtractionbyOpenIE extends Pipe {

	OpenIE openIE;
	
	StanfordCoreNLP pipeline;

	public RelationExtractionbyOpenIE() {
		openIE = new OpenIE(new ClearParser(new ClearPostagger(
				new ClearTokenizer(ClearTokenizer.defaultModelUrl()))),
				new ClearSrl(), false);
		
		Properties props = new Properties();
		props.put("annotators", "tokenize");
		pipeline = new StanfordCoreNLP(props);
	}

	public void debug() {

		String yy = "The suspect apparently called his wife from a cell phone shortly before the shooting began, saying he was acting out in revenge for something that happened 20 years ago, Miller said.";

		Seq<edu.knowitall.openie.Instance> xx = openIE.extract(yy);

		Iterator<edu.knowitall.openie.Instance> iteratorX = xx.iterator();
		while (iteratorX.hasNext()) {
			edu.knowitall.openie.Instance inst = iteratorX.next();
			Seq<Interval> offsets = inst.extr().arg1().offsets();
			Iterator<Interval> ii = offsets.iterator();
			while (ii.hasNext()) {
				Interval in = ii.next();
				int start = in.start();
				int end = in.end();
				System.out.println(yy.substring(start, end));
			}

			Seq<Interval> offsets2 = inst.extr().rel().offsets();
			Iterator<Interval> ii2 = offsets2.iterator();
			while (ii2.hasNext()) {
				Interval in = ii2.next();
				int start = in.start();
				int end = in.end();
				System.out.println(yy.substring(start, end));
			}
			
			Iterator<Argument> argIter = inst.extr().arg2s().iterator();
			while (argIter.hasNext()) {
				Argument arg2 = argIter.next();
				Seq<Interval> offsets3 = arg2.offsets();
				Iterator<Interval> ii3 = offsets3.iterator();
				while (ii3.hasNext()) {
					Interval in = ii3.next();
					int start = in.start();
					int end = in.end();
					System.out.println(yy.substring(start, end));
				}
			}
			
		}

	}
	
	// may not be continuous
	private edu.pengli.nlp.conference.acl2015.types.Argument getArgument(Argument arg1, 
			TreeMap<Integer, CoreLabel> positionCoreLabelMap, String originalSent, StanfordCoreNLP pipeline){
		
		String arg1Mention = arg1.text();
					
		Iterator<Interval> iiArg1 = arg1.offsets().iterator();
		int startPositionArg1 = -1;
		while(iiArg1.hasNext()) {
			Interval in = iiArg1.next();
			startPositionArg1 = in.start();
		}		
		edu.pengli.nlp.conference.acl2015.types.Argument Arg1 = 
				new edu.pengli.nlp.conference.acl2015.types.Argument();
			
		Arg1.add(positionCoreLabelMap.get(startPositionArg1));
		
		Annotation arg1Ann = new Annotation(arg1Mention);
		pipeline.annotate(arg1Ann);
		ArrayList<String> arg1Toks = new ArrayList<String>();
		for (CoreLabel token: arg1Ann.get(TokensAnnotation.class)){
			arg1Toks.add(token.originalText());
		}
	
		if(originalSent.contains(arg1Mention)){
			for (int i = 0; i < arg1Toks.size()-1; i++) {
				String argTok = arg1Toks.get(i);
				int start = startPositionArg1+argTok.length()+1;	
				CoreLabel lab = positionCoreLabelMap.get(start);
				if( lab == null){
					System.out.println("Argument sucks");
					System.exit(0);
				}
				Arg1.add(lab);
				startPositionArg1 += argTok.length()+1;
			}
			
		}else{
			
			for (int i = 1; i < arg1Toks.size(); i++) {
				String arg1Tok = arg1Toks.get(i);
				int start = originalSent.indexOf(" "+arg1Tok);
				CoreLabel lab = positionCoreLabelMap.get(start+1);
				if(lab == null){
					System.out.println("Argument sucks 2");
					System.exit(0);
				}
				Arg1.add(lab);
			}
			
		}
		return Arg1;
	}
	
	
	private edu.pengli.nlp.conference.acl2015.types.Predicate getRelation(Relation rel, 
			TreeMap<Integer, CoreLabel> positionCoreLabelMap, 
			String relMention, String originalSent, StanfordCoreNLP pipeline){
		
		if(relMention == null){
			relMention = rel.text();
		}
			
		Iterator<Interval> iiRel = rel.offsets().iterator();
		int startPositionRel = -1;
		if(iiRel.hasNext()) {
			Interval in = iiRel.next();
			startPositionRel = in.start();
		}

		edu.pengli.nlp.conference.acl2015.types.Predicate Rel = 
				new edu.pengli.nlp.conference.acl2015.types.Predicate();
		
		Rel.add(positionCoreLabelMap.get(startPositionRel));
		
		
		// prevent 're be separate by below 
		if(relMention.split(" ").length == 1){
			return Rel;
		}
		
		Annotation relAnn = new Annotation(relMention);
		pipeline.annotate(relAnn);
		ArrayList<String> relToks = new ArrayList<String>();
		for (CoreLabel token: relAnn.get(TokensAnnotation.class)){
			relToks.add(token.originalText());
			
		}
		
		if(originalSent.contains(relMention)){
		
			for (int i = 0; i < relToks.size()-1; i++) {
				String relTok = relToks.get(i);
				int start = startPositionRel+relTok.length()+1;
				
				CoreLabel lab = positionCoreLabelMap.get(start);
				if(lab == null){
					System.out.println("Relation sucks");
					System.exit(0);
				}
				Rel.add(lab);
				startPositionRel += relTok.length()+1;
		
			}
			
		}else{

			for (int i = 1; i < relToks.size(); i++) {
				String relTok = relToks.get(i);
				int start = originalSent.indexOf(" "+relTok);
				CoreLabel lab = positionCoreLabelMap.get(start+1);
				if(lab == null){
					System.out.println("Relation sucks 2");
					System.exit(0);
				}
				Rel.add(lab);
		
			}	
		}
		

		return Rel;
	}
	

	public Instance pipe(Instance instance) {

		Annotation document = (Annotation) instance.getData();
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);

		HashMap<CoreMap, ArrayList<Tuple>> map = new HashMap<CoreMap, ArrayList<Tuple>>();

		for (CoreMap sentence : sentences) {			
			TreeMap<Integer, CoreLabel> beginPositionCoreLabelMap = new TreeMap<Integer, CoreLabel>();
			//Using beginPosition due to openIE arguemnt and relation
			//can have offset. OpenIE don't have index. 
			int beginPosition = 0;
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			beginPositionCoreLabelMap.put(beginPosition, labels.get(0));
			StringBuilder sb = new StringBuilder();
			for(int i=0; i< labels.size()-1; i++){
				CoreLabel token = labels.get(i);
				sb.append(token.originalText()+" ");
				int range = token.originalText().length()+1;
				beginPosition += range;
				beginPositionCoreLabelMap.put(beginPosition, labels.get(i+1));
			}
			
			String sentenceMention = sb.toString().trim();
		
			Seq<edu.knowitall.openie.Instance> extractions = openIE
					.extract(sentenceMention);

			Iterator<edu.knowitall.openie.Instance> iterator = extractions
					.iterator();
			ArrayList<Tuple> tuples = new ArrayList<Tuple>();
			while (iterator.hasNext()) {
				edu.knowitall.openie.Instance inst = iterator.next();
				int itemSize = 2;
				Iterator<Argument> argiter = inst.extr().arg2s().iterator();
				while (argiter.hasNext()) {
					argiter.next();
					itemSize++;
				}
				double confidence = inst.confidence();
				//if there is no argument2
				if (itemSize == 2) {
					continue;
				}
				
				Argument arg1 = inst.extr().arg1();
					
				Relation rel = inst.extr().rel();
				String relMention = rel.text();
				if (relMention.matches(".*\\[.*?\\].*")) {
					continue;
				}
				
				edu.pengli.nlp.conference.acl2015.types.Argument Arg1 = 
						getArgument(arg1, beginPositionCoreLabelMap, sentenceMention, pipeline);
				
				if (itemSize == 3 || itemSize == 4) {
					
					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = 
							getRelation(rel, beginPositionCoreLabelMap, null, sentenceMention, pipeline);

					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					while (argIter.hasNext()) {
						
						Argument arg2 = argIter.next();	
						
						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = 
								getArgument(arg2, beginPositionCoreLabelMap, sentenceMention, pipeline);

						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}

				} else if (itemSize > 4) {
					
					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					ArrayList<Argument> arg2List = new ArrayList<Argument>();
					while (argIter.hasNext()) {
						Argument arg2 = argIter.next();
						arg2List.add(arg2);
					}
					String newRel = relMention + " " + arg2List.get(0).text();
					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = 
							getRelation(rel, beginPositionCoreLabelMap, newRel, 
									sentenceMention, pipeline);
					
					for (int i = 1; i < arg2List.size(); i++) {
				
						Argument arg2 = arg2List.get(i);
						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = 
								getArgument(arg2, beginPositionCoreLabelMap, 
										sentenceMention, pipeline);
								
						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}

				} else {
					System.out.println("Item size is wired");
					System.out.println(inst.toString());
					System.exit(0);
				}
			}
			map.put(sentence, tuples);
		}
		instance.setData(map);
		return instance;
	}

	// for testing
	public static void main(String[] args) {
		RelationExtractionbyOpenIE xx = new RelationExtractionbyOpenIE();
		xx.debug();
	}
}
