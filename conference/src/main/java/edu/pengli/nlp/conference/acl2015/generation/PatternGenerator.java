package edu.pengli.nlp.conference.acl2015.generation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import edu.pengli.nlp.conference.acl2015.pipe.FramenetTagger;
import edu.pengli.nlp.conference.acl2015.pipe.HeadAnnotation;
import edu.pengli.nlp.conference.acl2015.pipe.WordnetTagger;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.util.CoreMap;

public class PatternGenerator {
	
	FramenetTagger framenetTagger;
	WordnetTagger wordnetTagger;
	
	public PatternGenerator(FramenetTagger framenetTagger, WordnetTagger wordnetTagger){
		this.framenetTagger = framenetTagger;
		this.wordnetTagger = wordnetTagger;
	}
	
	public void run(String outputSummaryDir, String corpusName,
			InstanceList corpus, HeadAnnotation headAnnotator) throws Exception {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));

		corpus.readObject(in);
		in.close();

		HashSet<Pattern> patternSet = new HashSet<Pattern>();

		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" + corpusName + ".patterns.ser"));
		PrintWriter outt = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".tuples");
		PrintWriter outp = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".patterns");

		int docID = 0;
		for (Instance doc : corpus) {
			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();
			for (CoreMap sent : map.keySet()) {
				ArrayList<Tuple> tuples = map.get(sent);
				for (Tuple t : tuples) {

					ArrayList<IndexedWord> toks = new ArrayList<IndexedWord>();
					toks.addAll(t.getArg1());
					toks.addAll(t.getRel());
					toks.addAll(t.getArg2());

					// tuple fusion need know docID to compare IndexedWord
					for (IndexedWord iw : toks) {
						iw.setDocID(Integer.toString(docID));
					}

					if (t.getRel().lemmatext().equals("say"))
						continue;

					outt.println(t.getSentenceRepresentation());
					edu.pengli.nlp.conference.acl2015.types.Argument arg1 = headAnnotator
							.annotateArgHead(t.getArg1(), sent);
					t.setArg1(arg1);

					edu.pengli.nlp.conference.acl2015.types.Argument arg2 = headAnnotator
							.annotateArgHead(t.getArg2(), sent);
					t.setArg2(arg2);

					// for later sentence realization to get head verb
					edu.pengli.nlp.conference.acl2015.types.Predicate pre = headAnnotator
							.annotatePredicateHead(t.getRel(), sent);
					t.setRel(pre);

					// for complicated arguments, just ignore, so arg may be
					// null
					if (arg1 == null || arg2 == null)
						continue;

					// tuple with no head should not go into clustering ?
					if (pre.getHead() == null)
						continue;

					if (arg1.getHead() != null && arg2.getHead() != null) {

						// stanford NER tagger
						if (!arg1.getHead().ner().equals("O")
								&& !arg2.getHead().ner().equals("O")) {

							Pattern p = new Pattern(arg1, pre, arg2, t);
							patternSet.add(p);

						} else {

							wordnetTagger.annotatePerson(arg1, arg2);
							framenetTagger.annotate(arg1, pre, arg2);

							if (arg1.getHead().ner().equals("O")
									|| arg2.getHead().ner().equals("O")) {
								wordnetTagger.annotateNoun(arg1, arg2, t);
							}

							if (!arg1.getHead().ner().equals("O")
									&& !arg2.getHead().ner().equals("O")) {
								Pattern p = new Pattern(arg1, pre, arg2, t);
								patternSet.add(p);
								outp.println(p.toString());
							}

						}

					}
				}
			}

			docID++;
		}

		outt.close();
		out.writeObject(patternSet);
		out.close();
		outp.close();
	}
	

}
