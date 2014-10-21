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
import edu.pengli.nlp.platform.algorithms.miscellaneous.LongestCommonSubstring;
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
	private edu.pengli.nlp.conference.acl2015.types.Argument getArgument(
			Argument arg, TreeMap<Integer, CoreLabel> positionCoreLabelMap,
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
		

		Arg.add(positionCoreLabelMap.get(startPositionArg));

		Annotation argAnn = new Annotation(argMention);
		pipeline.annotate(argAnn);
		ArrayList<String> argToks = new ArrayList<String>();
		for (CoreLabel token : argAnn.get(TokensAnnotation.class)) {
			argToks.add(token.originalText());
		}

		
		int flagPosition = startPositionArg;	
		if (originalSent.contains(argMention)) {
			for (int i = 0; i < argToks.size() - 1; i++) {
				String argTok = argToks.get(i);
				int start = flagPosition + argTok.length() + 1;
				CoreLabel lab = positionCoreLabelMap.get(start);
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
				CoreLabel lab = positionCoreLabelMap.get(start);
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
			Relation rel, TreeMap<Integer, CoreLabel> positionCoreLabelMap,
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

		Rel.add(positionCoreLabelMap.get(startPositionRel));
		
		Annotation relAnn = new Annotation(relMention);
		pipeline.annotate(relAnn);
		ArrayList<String> relToks = new ArrayList<String>();
		for (CoreLabel token : relAnn.get(TokensAnnotation.class)) {
			relToks.add(token.originalText());

		}

		// prevent 're be separate by below
		if (relMention.split(" ").length == 1) {
			return Rel;
		}

		int flagPosition = startPositionRel;	
		if (originalSent.contains(relMention)) {
	
			for (int i = 0; i < relToks.size() - 1; i++) {
				String relTok = relToks.get(i);
				int start = flagPosition + relTok.length() + 1;

				CoreLabel lab = positionCoreLabelMap.get(start);
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
				CoreLabel lab = positionCoreLabelMap.get(start);
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
			TreeMap<Integer, CoreLabel> beginPositionCoreLabelMap = new TreeMap<Integer, CoreLabel>();
			// Using beginPosition due to openIE arguemnt and relation
			// can have offset. OpenIE don't have index.
			int beginPosition = 0;
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			beginPositionCoreLabelMap.put(beginPosition, labels.get(0));
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < labels.size() - 1; i++) {
				CoreLabel token = labels.get(i);
				sb.append(token.originalText() + " ");
				int range = token.originalText().length() + 1;
				beginPosition += range;
				beginPositionCoreLabelMap.put(beginPosition, labels.get(i + 1));
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
				// if there is no argument2
				if (itemSize == 2) {
					continue;
				}

				Argument arg1 = inst.extr().arg1();

				Relation rel = inst.extr().rel();
				String relMention = rel.text();
				if (relMention.matches(".*\\[.*?\\].*")) {
					continue;
				}

				edu.pengli.nlp.conference.acl2015.types.Argument Arg1 = getArgument(
						arg1, beginPositionCoreLabelMap, sentenceMention,
						pipeline);

				if (itemSize == 3 || itemSize == 4) {

					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = getRelation(
							rel, beginPositionCoreLabelMap, null,
							sentenceMention, pipeline);

					Iterator<Argument> argIter = inst.extr().arg2s().iterator();
					while (argIter.hasNext()) {

						Argument arg2 = argIter.next();

						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = getArgument(
								arg2, beginPositionCoreLabelMap,
								sentenceMention, pipeline);

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
					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = getRelation(
							rel, beginPositionCoreLabelMap, newRel,
							sentenceMention, pipeline);

					for (int i = 1; i < arg2List.size(); i++) {

						Argument arg2 = arg2List.get(i);
						edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = getArgument(
								arg2, beginPositionCoreLabelMap,
								sentenceMention, pipeline);

						Tuple t = new Tuple(confidence, Arg1, Rel, Arg2);
						tuples.add(t);
					}

				}
			}
			
			for(Tuple t : tuples){
				edu.pengli.nlp.conference.acl2015.types.Argument Arg2 = t.getArg2();
				ArrayList<CoreLabel> prep = findPrePhrase(Arg2, sentence);
				if(prep != null){
					edu.pengli.nlp.conference.acl2015.types.Predicate Rel = t.gerRel();
					Rel.addAll(prep);			
					t.setRel(Rel);
					for(CoreLabel p : prep){
						Arg2.remove(p);
					}
					t.setArg2(Arg2);
				}	
			}	
			map.put(sentence, tuples);
		}
		instance.setData(map);
		return instance;
	}
	
	private ArrayList<CoreLabel> findPrePhrase(edu.pengli.nlp.conference.acl2015.types.Argument 
			Arg, CoreMap sent){
		
		SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);
		
		if(Arg.size() == 1)
			return null;
		
		ArrayList<CoreLabel> ret = null;
		
		for(int i=0; i<Arg.size(); i++){
			CoreLabel tok = Arg.get(i);
			if(tok == null){
				return null;
			}
			IndexedWord iw = graph.getNodeByIndexSafe(tok.index());
			if(iw == null){// some punctuation don't map to the graph 
				continue;
			}
			Iterable<SemanticGraphEdge> edges = graph.outgoingEdgeIterable(iw);
			for(SemanticGraphEdge e : edges){
				GrammaticalRelation gr = e.getRelation();
				if (gr.toString().equals("pobj") || gr.toString().equals("pcomp")) {
					ret = new ArrayList<CoreLabel>();
					for(int j=0; j<=i; j++){
						ret.add(Arg.get(j));
					}
					return ret;
				}
				if(gr.toString().equals("nsubj")){
					return null;
				}
			}
		}
		
		return ret;
	}

	// for testing
	public static void main(String[] args) {
		RelationExtractionbyOpenIE xx = new RelationExtractionbyOpenIE();
		xx.debug();
	}
}
