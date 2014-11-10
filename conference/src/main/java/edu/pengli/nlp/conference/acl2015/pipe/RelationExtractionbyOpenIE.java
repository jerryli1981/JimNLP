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
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.trees.GrammaticalRelation;
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

	private void debug() {

		String yy = "The suspect apparently called his wife from a cell phone shortly before the shooting began, saying he was acting out in revenge for something that happened 20 years ago, Miller said.";

		Seq<edu.knowitall.openie.Instance> xx = openIE.extract(yy);

		Iterator<edu.knowitall.openie.Instance> iteratorX = xx.iterator();
		while (iteratorX.hasNext()) {
			edu.knowitall.openie.Instance inst = iteratorX.next();
			System.out.print("["+inst.extr().arg1().text()+"]");
			System.out.print("["+inst.extr().rel().text()+"]");

			Iterator<Argument> argIter = inst.extr().arg2s().iterator();
			while (argIter.hasNext()) {
				
				Argument arg2 = argIter.next();
				System.out.print("["+arg2.text()+"]");
			}
			
			System.out.println();

		}

	}

	// may not be continuous
	private edu.pengli.nlp.conference.acl2015.types.Argument getArgument(
			Argument arg, TreeMap<Integer, IndexedWord> positionWordMap,
			String originalSent, StanfordCoreNLP pipeline) {

		String argMention = arg.text();
		Iterator<Interval> iiArg = arg.offsets().iterator();
		int startPositionArg = -1;
		if(iiArg.hasNext()) {
			Interval in = iiArg.next();
			startPositionArg = in.start();
		}
		edu.pengli.nlp.conference.acl2015.types.Argument Arg = 
				new edu.pengli.nlp.conference.acl2015.types.Argument();
		
		Annotation argAnn = new Annotation(argMention);
		
		pipeline.annotate(argAnn);
		ArrayList<String> argToks = new ArrayList<String>();
		for (CoreLabel token : argAnn.get(TokensAnnotation.class)) {
			argToks.add(token.originalText());
		}
		
		IndexedWord st = positionWordMap.get(startPositionArg);
		
		if(st == null){
			for(int posi : positionWordMap.keySet()){
				IndexedWord tok = positionWordMap.get(posi);
				if(tok == null)
					continue;
				if(tok.originalText().contains(argToks.get(0))){
					char[] cs = tok.originalText().toCharArray();
					boolean containsPunc = false;
					for(char c : cs){
						StringBuilder sb = new StringBuilder();
						sb.append(c);
						String charMention = sb.toString();
						if(charMention.matches("\\p{Punct}"))
							containsPunc = true;
					}
					if(containsPunc == true){
						startPositionArg = posi;
						st = tok;
						break;
					}
				}
			}
		}
		
		if(st == null){
			System.out.println("Argument 3 sucks");
			System.exit(0);
		}
        
		Arg.add(st);

		int flagPosition = startPositionArg;	
		if (originalSent.contains(argMention)) {
			for (int i = 0; i < argToks.size() - 1; i++) {
				String argTok = argToks.get(i);
				int start = flagPosition + argTok.length() + 1;
				IndexedWord lab = positionWordMap.get(start);
				if (lab == null) {
					System.out.println("Argument sucks");
					System.exit(0);
				}
				Arg.add(lab);
				flagPosition += argTok.length() + 1;
			}

		} else {
			
			String subSentence = originalSent.substring(flagPosition);
			
			for (int i = 1; i < argToks.size(); i++) {
				String argTok = argToks.get(i);
				int start = subSentence.indexOf(" " + argTok)+flagPosition+1;
				IndexedWord lab = positionWordMap.get(start);
				if (lab == null) {
					System.out.println("Argument 2 sucks");
					System.exit(0);
				}
				Arg.add(lab);

			}
		}
		return Arg;
	}

	private edu.pengli.nlp.conference.acl2015.types.Predicate getRelation(
			Relation rel, TreeMap<Integer, IndexedWord> positionWordMap,
			String relMention, String originalSent, StanfordCoreNLP pipeline) {

		if (relMention == null) {
			relMention = rel.text();
		}

		Iterator<Interval> iiRel = rel.offsets().iterator();
		int startPositionRel = -1;
		if (iiRel.hasNext()) {
			Interval in = iiRel.next();
			startPositionRel = in.start();
		}

		edu.pengli.nlp.conference.acl2015.types.Predicate Rel = 
				new edu.pengli.nlp.conference.acl2015.types.Predicate();
		
		Annotation relAnn = new Annotation(relMention);
		pipeline.annotate(relAnn);
		ArrayList<String> relToks = new ArrayList<String>();
		for (CoreLabel token : relAnn.get(TokensAnnotation.class)) {
			relToks.add(token.originalText());

		}
        //us-led, stanford don't separate, however openIE separate
		IndexedWord st = positionWordMap.get(startPositionRel);
		if(st == null){
			for(int posi : positionWordMap.keySet()){
				IndexedWord tok = positionWordMap.get(posi);
				if(tok == null)
					continue;
				if(tok.originalText().contains(relToks.get(0))){
					char[] cs = tok.originalText().toCharArray();
					boolean containsPunc = false;
					for(char c : cs){
						StringBuilder sb = new StringBuilder();
						sb.append(c);
						String charMention = sb.toString();
						if(charMention.matches("\\p{Punct}"))
							containsPunc = true;
					}
					if(containsPunc == true){
						startPositionRel = posi;
						st = tok;
						break;
					}
				}
			}
		}
		
		if(st == null){
			System.out.println("Relation 3 sucks");
			System.exit(0);
		}
		
		Rel.add(st);
	
		// prevent 're be separate by below
		if (relMention.split(" ").length == 1) {
			return Rel;
		}

		int flagPosition = startPositionRel;	
		if (originalSent.contains(relMention)) {
	
			for (int i = 0; i < relToks.size() - 1; i++) {
				String relTok = relToks.get(i);
				int start = flagPosition + relTok.length() + 1;

				IndexedWord lab = positionWordMap.get(start);
				if (lab == null) {
					System.out.println("Relation sucks");
					System.exit(0);
				}
				Rel.add(lab);
				flagPosition += relTok.length() + 1;

			}

		} else {
			
			String subSentence = originalSent.substring(flagPosition);
			
			for (int i = 1; i < relToks.size(); i++) {
				String relTok = relToks.get(i);
				int start = subSentence.indexOf(" " + relTok)+flagPosition+1;
				IndexedWord lab = positionWordMap.get(start);
				if (lab == null) {
					System.out.println("Relation 2 sucks");
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
			TreeMap<Integer, IndexedWord> beginPositionWordMap = new TreeMap<Integer, IndexedWord>();

			// Using beginPosition due to openIE arguemnt and relation
			// can have offset. OpenIE don't have index.

			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			int beginPosition = 0;
			beginPositionWordMap.put(beginPosition, graph.getNodeByIndexSafe(labels.get(0).index()));
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < labels.size() - 1; i++) {
				CoreLabel token = labels.get(i);
				sb.append(token.originalText() + " ");
				CoreLabel nextToken = labels.get(i+1);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(nextToken.index());
				int range = token.originalText().length() + 1;
				beginPosition += range;
				if(word != null)
					beginPositionWordMap.put(beginPosition, word);
				else{
					IndexedWord punc = new IndexedWord(nextToken);
					beginPositionWordMap.put(beginPosition, punc);
				}
			}
								
			//here need use tokenized sentence to input openIE to keep beginPosition mapping correct.
			String sentenceMention = sb.toString().trim();
			Seq<edu.knowitall.openie.Instance> extractions = openIE
					.extract(sentenceMention);

			Iterator<edu.knowitall.openie.Instance> iterator = extractions
					.iterator();
			ArrayList<Tuple> tuples = new ArrayList<Tuple>();
			while (iterator.hasNext()) {
				edu.knowitall.openie.Instance inst = iterator.next();
				
				Iterator<Argument> argiter = inst.extr().arg2s().iterator();
				if(!argiter.hasNext())// if there is no argument2
					continue;
				
				int arg2ItemSize = 0;
				while (argiter.hasNext()) {
					argiter.next();
					arg2ItemSize++;
				}
				
				double confidence = inst.confidence();
				
				Argument arg1 = inst.extr().arg1();

				Relation rel = inst.extr().rel();
				String relMention = rel.text();
				if (relMention.matches(".*\\[.*?\\].*"))
					continue;
				
				edu.pengli.nlp.conference.acl2015.types.Argument Arg1 = getArgument(
						arg1, beginPositionWordMap, sentenceMention,
						pipeline);
				

				if (arg2ItemSize == 1 || arg2ItemSize == 2) {

					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = getRelation(
							rel, beginPositionWordMap, null,
							sentenceMention, pipeline);

					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					while (argIter.hasNext()) {

						Argument arg2 = argIter.next();

						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = getArgument(
								arg2, beginPositionWordMap,
								sentenceMention, pipeline);
						
						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}

				} else if (arg2ItemSize > 2) {

					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					ArrayList<Argument> arg2List = new ArrayList<Argument>();
					while (argIter.hasNext()) {
						Argument arg2 = argIter.next();
						arg2List.add(arg2);
					}
					String newRel = relMention + " " + arg2List.get(0).text();
					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = getRelation(
							rel, beginPositionWordMap, newRel,
							sentenceMention, pipeline);
					
					for (int i = 1; i < arg2List.size(); i++) {

						Argument arg2 = arg2List.get(i);
						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = getArgument(
								arg2, beginPositionWordMap,
								sentenceMention, pipeline);

						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}
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
