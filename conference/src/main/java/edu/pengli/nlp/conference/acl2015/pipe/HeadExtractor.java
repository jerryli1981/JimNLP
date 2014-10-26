package edu.pengli.nlp.conference.acl2015.pipe;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import edu.pengli.nlp.conference.acl2015.types.Argument;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class HeadExtractor {

	static HeadFinder headFinder;

	public HeadExtractor() {
		headFinder = new CollinsHeadFinder();
	}

	public Argument extract(Argument arg, CoreMap sent) {

		if (arg.size() == 1)
			return arg;

		Tree tree = sent.get(TreeAnnotation.class);
		HashMap<String, String> npheadMap = new HashMap<String, String>();
		dfs(tree, tree, headFinder, npheadMap);

		String argMention = arg.toString();
		boolean find = false;
		Argument head = null;
		for (String np : npheadMap.keySet()) {
			if(npheadMap.containsKey(argMention)){
				find  = true;
				String headMention = npheadMap.get(np);
				for (CoreLabel lab : arg) {
					if (lab.originalText().equals(headMention)) {
						head = new Argument();
						head.add(lab);
					}
				}
			}
		}
		
		if(find == true)
			return head;
		
		else{
			
			SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);
			HashMap<CoreLabel, IndexedWord> iwordCoreLabelMap = new HashMap<CoreLabel, IndexedWord>();	
			for (CoreLabel token : sent.get(TokensAnnotation.class)) {
				int coreLabelIdx = token.index();
				IndexedWord iw = graph.getNodeByIndexSafe(coreLabelIdx);
				if(iw == null){ // some punctuation don't map to the graph 
					continue;
				}else{
					iwordCoreLabelMap.put(token, iw);
				}
			}
		
			ArrayList<IndexedWord[]> pairs = new ArrayList<IndexedWord[]>();
			for (int i = 0; i < arg.size(); i++) {
				for (int j = 0; j < arg.size(); j++) {
					if (i == j)
						continue;
					else {
						IndexedWord[] pair = new IndexedWord[2];
						IndexedWord ai = iwordCoreLabelMap.get(arg.get(i));
						IndexedWord aj = iwordCoreLabelMap.get(arg.get(j));
						pair[0] = ai;
						pair[1] = aj;
						pairs.add(pair);
					}
				}
			}
			
			List<SemanticGraphEdge> path = new ArrayList<SemanticGraphEdge>();		
			for (IndexedWord[] pa : pairs) {
				List<SemanticGraphEdge> edge = graph.getAllEdges(pa[0], pa[1]);
				path.addAll(edge);
			}
			
			if (containsSubVerbOrVerbObj(path) == true)
				return null;
			
			head = decideArgHead(arg, sent, path);
			if(head != null)
				return head;
			else
				return null;
		}
		
	}

	private int getIndex(Argument arg, IndexedWord tok) {
		for (int i = 0; i < arg.size(); i++) {
			if (arg.get(i).originalText().equals(tok.originalText()))
				return i;
		}
		return -1;
	}

	private boolean containsSubVerbOrVerbObj(List<SemanticGraphEdge> path) {
		
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("nsubj") || gr.toString().equals("ccomp")
					|| gr.toString().equals("nsubjpass")) {
				return true;
			}
		}
		return false;
	}

	private Argument decideArgHead(Argument arg, CoreMap sent,
			List<SemanticGraphEdge> path) {
		
		int startNodeIdx = arg.get(0).index();
		int endNodeIdx = arg.get(arg.size()-1).index();
		
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("prep")) {
				IndexedWord dep = edge.getDependent();
				if (dep.index() != startNodeIdx){
					endNodeIdx = dep.index();
					break;
				}
					
			}

		}
		
		Argument head = null;
		boolean find = false;
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("det") || gr.toString().equals("amod")
					|| gr.toString().equals("poss")
					|| gr.toString().equals("num")
					|| gr.toString().equals("advmod")
					|| gr.toString().equals("prep")) {
				IndexedWord gov = edge.getGovernor();
				if (!gov.tag().startsWith("NN"))
					continue;
				if (gov.index() < startNodeIdx
						|| gov.index() > endNodeIdx)
					continue;
				
				if (getIndex(arg, gov) == -1) {
					continue;
				}
				head = new Argument();
				head.add(arg.get(getIndex(arg, gov)));
				find = true;
			} else if (gr.toString().equals("tmod")|| gr.toString().equals("nn")) {
				IndexedWord dep = edge.getDependent();
				if (!dep.tag().startsWith("NN"))
					continue;
				if (dep.index() < startNodeIdx
						|| dep.index() > endNodeIdx)
					continue;
				
				if (getIndex(arg, dep) == -1) {
					continue;
				}
				
				head = new Argument();
				head.add(arg.get(getIndex(arg, dep)));
				find = true;
			}
		}
		
		boolean findNNS = false;
		if(find == true)
			return head;
		else{
			for (int i = 0; i < arg.size(); i++) {
				if (arg.get(i).tag().equals("NNS")) {
					head = new Argument();
					head.add(arg.get(i));
					findNNS = true;
				}
			}
		}
		
		if(findNNS == true)
			return head;
		else
			return null;
	}

	public static void dfs(Tree node, Tree parent, HeadFinder headFinder,
			HashMap<String, String> map) {
		if (node == null || node.isLeaf()) {
			return;
		}
		// if node is a NP - Get the terminal nodes to get the words in the NP
		if (node.value().equals("NP")) {

			// System.out.println(" Noun Phrase is ");
			List<Tree> leaves = node.getLeaves();
			StringBuilder np = new StringBuilder();
			for (Tree leaf : leaves) {
				// System.out.print(leaf.toString()+" ");
				np.append(leaf.toString() + " ");
			}

			// System.out.println();
			// System.out.println(" Head string is ");
			// System.out.println(node.headTerminal(headFinder, parent));
			String head = node.headTerminal(headFinder, parent).toString();
			String nounPhrase = np.toString().trim();
			nounPhrase = nounPhrase.replaceAll("\\s,", ",");
			nounPhrase = nounPhrase.replaceAll(" '", "'");

			map.put(nounPhrase, head);

		}
		for (Tree child : node.children()) {
			dfs(child, node, headFinder, map);
		}
	}

}
