package edu.pengli.nlp.platform.pipe;

import java.util.List;
import java.util.Properties;

import edu.pengli.nlp.platform.types.Instance;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class SentenceParsing extends Pipe{
	
	AnnotationPipeline parser;

	public SentenceParsing() {

		Properties props = new Properties();
		props.put("annotators", "tokenize, ssplit, parse");
		props.put("parse.model", "../models/Stanford/models/lexparser/englishPCFG.ser.gz");
		props.put("parser.flags", "-retainTmpSubcategories");
		props.put("pos.model", "../models/Stanford/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
		parser = new StanfordCoreNLP(props);
	}

	public Instance pipe(Instance inst) {
		String text = (String) inst.getData();
		Annotation annotation = new Annotation(text);
		parser.annotate(annotation);
		List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
		if(sentences.size() > 1){
			System.out.println("Impossible of Sentence Parsing");
			System.exit(0);
		}
		Tree tree = null;
		SemanticGraph dependencies = null;
		for(CoreMap sentence: sentences) {
		   tree = sentence.get(TreeAnnotation.class);
		   dependencies = sentence.get(BasicDependenciesAnnotation.class);
		}
		inst.setData(dependencies);		
		return inst;
	}

}
