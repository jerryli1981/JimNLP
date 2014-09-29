package edu.pengli.nlp.conference.acl2015.generation;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.conference.acl2015.types.InformationItem;
import edu.pengli.nlp.conference.acl2015.types.NewsCorpus;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.DependencyGraph;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.Dependency;
import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeGraphNode;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.english.Realiser;

public class AbstractiveGeneration {

	NLGFactory nlgFactory;
	Realiser realiser;
	LexicalizedParser lp;

	public AbstractiveGeneration(LexicalizedParser lp) {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
		this.lp = lp;

	}

	private Tree parseSentence(Instance sent) {

		TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(
				new CoreLabelTokenFactory(), "");
		Tokenizer<CoreLabel> tok = tokenizerFactory
				.getTokenizer(new StringReader((String) sent.getSource()));
		List<CoreLabel> rawWords2 = tok.tokenize();

		return lp.apply(rawWords2);

	}

	private ArrayList<InformationItem> extractInformationItems(Tree parseTree, GrammaticalStructure gs) {
		Collection<TypedDependency> tds = gs.typedDependencies();
		DependencyGraph graph = new DependencyGraph(tds.size()*2);

		for (TypedDependency td : tds) {
			graph.addEdge(td);
		}
		
		HashSet<TreeGraphNode> predicates = new HashSet<TreeGraphNode>();
		TreeGraphNode subjectIncase = null;
		for (TypedDependency td : tds) {
			TreeGraphNode gov = td.gov();
			GrammaticalRelation gr = td.reln();
			if (gr.toString().equals("nsubj") || gr.toString().equals("dobj") 
					|| (gr.toString().equals("prep") && gov.parent().nodeString().startsWith("VB"))) {
				predicates.add(gov);
			}
			
			if (gr.toString().equals("nsubj")){
				Iterable<TypedDependency> iter = graph.adj(td.gov().index());
				for(TypedDependency child : iter){
					GrammaticalRelation dgr = child.reln(); 
					if(dgr.toString().equals("dobj") || 
							(dgr.toString().equals("prep") && 
									child.gov().parent().nodeString().startsWith("VB"))){
						subjectIncase = td.dep();
					}
					
				}
				
			}
		}

		ArrayList<InformationItem> possibleItems = new ArrayList<InformationItem>();

		for (TreeGraphNode p : predicates) {
			
			boolean subjectExist = false;
			boolean directObjectExist = false;
			boolean prepObjectExist = false;
			TreeGraphNode subject = null;
			TreeGraphNode directObject = null;
			TreeGraphNode prep = null;
			TreeGraphNode prepObject = null;
			
			for (TypedDependency td : tds) {
				TreeGraphNode gov = td.gov();
				if(!gov.equals(p)) continue;
				TreeGraphNode dep = td.dep();
				GrammaticalRelation gr = td.reln();
				if (gr.toString().equals("nsubj")) {
					subjectExist = true;
					subject = dep;
				}
				if (gr.getShortName().equals("dobj")) {
					directObjectExist = true;
					directObject = dep;
				}
				if(gr.toString().equals("prep") && gov.parent().nodeString().startsWith("VB")){
					
					for (TypedDependency pair : tds) {
						TreeGraphNode g = pair.gov();
						if(dep.equals(g) && pair.reln().toString().equals("pobj")){
							prepObjectExist = true;
							prep = g;
							prepObject = pair.dep();
						}
						
					}
				}
			}
			
			if(subjectExist == false && directObjectExist ==true && prepObjectExist == false){
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
                obj.add(directObject);
				possibleItems.add(new InformationItem(subjectIncase, p, obj));	
				
			}else if(subjectExist == false && directObjectExist ==false && prepObjectExist == true){
				
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
                obj.add(prep);
                obj.add(prepObject);
				possibleItems.add(new InformationItem(subjectIncase, p, obj));
				
			}else if(subjectExist == true && directObjectExist ==false && prepObjectExist == false){

				possibleItems.add(new InformationItem(subject, p, null));
				
			}else if(subjectExist == true && directObjectExist == true && prepObjectExist == false){
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
                obj.add(directObject);
				possibleItems.add(new InformationItem(subject, p, obj));	
			}else if(subjectExist == true && directObjectExist == false && prepObjectExist == true){
				
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
                obj.add(prep);
                obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));		
			}

		}

		return possibleItems;

	}

	private ArrayList<String> generate(Instance sent) {
		
		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		
		Tree parsedTree = parseSentence(sent);
		GrammaticalStructure gs = gsf.newGrammaticalStructure(parsedTree);
		Collection<TypedDependency> tds = gs.typedDependencies();

		
		SPhraseSpec newSent = nlgFactory.createClause();
		ArrayList<String> comSents = new ArrayList<String>();
		ArrayList<InformationItem> items = extractInformationItems(parsedTree, gs);

		System.out.println("Original Sent is: " + sent.getSource());
		if (items.size() != 0)
			for (InformationItem item : items) {

				System.out.println("Information Item is: " + item.toString());

				NPPhraseSpec subjectNp = generateNP(tds, item.getSubject());

				// System.out.println(lt.getRealiser().realiseSentence(subjectNp));

				newSent.setSubject(subjectNp);

				VPPhraseSpec vp = generateVP(tds, item.getPredicate(),
						item.getObject());

				newSent.setVerbPhrase(vp);

				String output = realiser.realiseSentence(newSent);

				System.out.println("Generated sent is: " + output);

				comSents.add(output);
			}
		return comSents;
	}

	private NPPhraseSpec generateNP(Collection<TypedDependency> tds, TreeGraphNode head) {
		
		HashSet<TreeGraphNode> set = new HashSet<TreeGraphNode>();
		for (TypedDependency td : tds) {
			set.add(td.dep());
			set.add(td.gov());
		}

		DependencyGraph graph = new DependencyGraph(set.size() * 2);

		for (TypedDependency td : tds) {
			graph.addEdge(td);
		}
		
		NPPhraseSpec np = nlgFactory.createNounPhrase();
		np.setHead(head.headWordNode().value());
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = head.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			for (TypedDependency td : iter) {
				if (td.gov().index() == td.dep().index())
					continue; // prevent infitive recusion
				int depIdx = td.dep().index();
				if (td.reln().toString().equals("prep")) {
					String prep = td.dep().nodeString();
					Iterable<TypedDependency> children = graph.adj(td.dep().index());
					TreeGraphNode obj = null;
					for(TypedDependency child : children){
						GrammaticalRelation dgr = child.reln(); 
						if(dgr.toString().equals("pobj")){
							obj = child.dep();
						}
					}
					PPPhraseSpec ppp = generatePrepP(tds, prep, obj);
					if (np.getPostModifiers().size() != 0) {
						np.addPostModifier(ppp);
					} else
						np.setPostModifier(ppp);
					continue; // do not deep travel any more
				} else if (td.reln().toString().equals("nn")) {
					NPPhraseSpec tmp = generateNP(tds, td.dep());
					if (td.dep().index() < head.index())
						np.setPreModifier(tmp);
					else
						np.setPostModifier(tmp);
					continue;
				} else if (td.reln().toString().startsWith("conj")) {
					NPPhraseSpec tmp = generateNP(tds, td.dep());
					String conj = td.reln().toString().replace("conj_", "");
					np.addPostModifier(conj + " ");
					np.addPostModifier(tmp);
					continue;
				} else if (td.reln().toString().equals("det")) {
					TreeGraphNode det = td.dep();

					np.setSpecifier(det.value());
				} else if (td.reln().toString().equals("num")) {
					TreeGraphNode numMod = td.dep();
					np.addPreModifier(numMod.value());

				} else if (td.reln().toString().equals("amod")) {
					TreeGraphNode adjMod = td.dep();
					if (adjMod.index() < head.index())
						np.addPreModifier(adjMod.value());
					else
						np.addPostModifier(adjMod.value());
				} else
					continue; // this is ignore all the other children

				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}

		return np;
	}

	// follow the order to set vp
	private VPPhraseSpec generateVP(Collection<TypedDependency> tds,
			TreeGraphNode headVp, ArrayList<TreeGraphNode> object) {
		
		DependencyGraph graph = new DependencyGraph(tds.size()*2);

		for (TypedDependency td : tds) {
			graph.addEdge(td);
		}

		VPPhraseSpec vp = nlgFactory.createVerbPhrase();
		vp.setHead(headVp.nodeString());


		if (object != null) {
			
			if(object.size() == 1){
				
				// set direct object
				NPPhraseSpec dirObjNp = generateNP(tds, object.get(0));
				vp.setObject(dirObjNp);

				// set indirect object from direct children
				Iterable<TypedDependency> iter = graph.adj(headVp.index());
				for (TypedDependency td : iter) {
					if (td.reln().toString().startsWith("iobj")) {
						NPPhraseSpec indirObjNp = generateNP(tds, td.dep());
						vp.setIndirectObject(indirObjNp);
						break;
					}
				}
				
			}
			
			if((object.size() == 2)){
				// set prep object from direct children
				Iterable<TypedDependency> iter = graph.adj(object.get(0).index());
				for (TypedDependency td : iter) {
					if (td.reln().toString().startsWith("pobj")) {
						PPPhraseSpec ppp = generatePrepP(tds, td.gov().nodeString(),td.dep());
						vp.setPostModifier(ppp);
						break;
					}
				}
			}

		} else {
			// set verb complement
			Iterable<TypedDependency> iter = graph.adj(headVp.index());
			for (TypedDependency td : iter) {
				if (td.reln().toString().startsWith("ccomp")
						&& (td.dep().index() > headVp.index())) {
					String comp = generateClauseComplement(graph, td.dep());
					if (vp.getPostModifiers().size() != 0) {
						vp.addPostModifier(comp);
					} else
						vp.setPostModifier(comp);
				}
			}

		}

/*		// set prep and open clause complement if it has
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = headVp.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			for (TypedDependency td : iter) {
				int depIdx = td.dep().index();
				if (td.reln().toString().startsWith("prep")) {
					String prep = null;
					if (td.reln().toString().equals("prep")) {
						continue;
					}
					prep = td.reln().toString().replaceAll("prepc?_", ""); // dependency
																			// has
																			// prepc
					prep = prep.replaceAll("_", " "); // prep_out_of
					TreeGraphNode obj = td.dep();
					PPPhraseSpec ppp = generatePrepP(tds, prep, obj);
					if (ppp == null)
						continue;
					if (vp.getPostModifiers().size() != 0) {
						vp.addPostModifier(ppp);
					} else
						vp.setPostModifier(ppp);
				} else if (td.reln().toString().startsWith("xcomp")) {
					String comp = generateClauseComplement(graph, td.dep());
					if (vp.getPostModifiers().size() != 0) {
						vp.addPostModifier(comp);
					} else
						vp.setPostModifier(comp);

				} else
					continue;// this is ignore all the other children

				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}

		}*/
		return vp;
	}

	private PPPhraseSpec generatePrepP(Collection<TypedDependency> tds, String prep,
			TreeGraphNode np) {
		PPPhraseSpec ppp = nlgFactory.createPrepositionPhrase();
		ppp.setPreposition(prep);
		NPPhraseSpec npp = generateNP(tds, np);
		ppp.setObject(npp);
		return ppp;
	}

	private String generateClauseComplement(DependencyGraph graph,
			TreeGraphNode predicate) {

		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = predicate.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		TreeMap<Integer, TreeGraphNode> map = new TreeMap<Integer, TreeGraphNode>();
		map.put(predicate.index(), predicate);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			for (TypedDependency td : iter) {
				int depIdx = td.dep().index();
				if (!marked[depIdx]) {
					map.put(depIdx, td.dep());
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}
		StringBuilder sb = new StringBuilder();
		Set<Integer> keys = map.keySet();
		for (Integer i : keys) {
			sb.append(map.get(i).value() + " ");
		}

		return sb.toString().trim();
	}

	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName) {

		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);

		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		NewsCorpus corpus = new NewsCorpus(fIter, pipeLine);

		pipeLine = new PipeLine();
		pipeLine.addPipe(new Noop());
		NewsCorpus docs = new NewsCorpus(corpus, pipeLine);

		InstanceList totalSentenceList = new InstanceList(null);
		for (Instance doc : docs) {
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				totalSentenceList.add(sent);
			}
		}

		for (Instance sent : totalSentenceList) {
			ArrayList<String> candidateSents = generate(sent);

		}

	}

}
