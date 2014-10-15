package edu.pengli.nlp.conference.acl2015.generation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.Stack;

import org.apache.commons.httpclient.DefaultHttpMethodRetryHandler;
import org.apache.commons.httpclient.Header;
import org.apache.commons.httpclient.HttpClient;
import org.apache.commons.httpclient.HttpException;
import org.apache.commons.httpclient.HttpMethod;
import org.apache.commons.httpclient.HttpStatus;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.params.HttpMethodParams;
import org.apache.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import edu.knowitall.openie.Argument;
import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.conference.acl2015.pipe.RelationExtraction;
import edu.pengli.nlp.conference.acl2015.types.InformationItem;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.pipe.CharSequenceCoreNLPAnnotation;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerLineIterator;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RankMap;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.util.CoreMap;
import scala.collection.Seq;
import scala.collection.Iterator;
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

	private final static String API_URL = "http://spotlight.dbpedia.org/";
	private static final double CONFIDENCE = 0.0;
	private static final int SUPPORT = 0;
	private static HttpClient client;
	private Logger LOG;

	public AbstractiveGeneration() {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
		LOG = Logger.getLogger(this.getClass());
		client = new HttpClient();
	}

	/*
	 * current implementation just cove direct object and prep object, subject
	 * and predicate are necessary. object could be empty.
	 */

	private ArrayList<InformationItem> extractInformationItems(
			SemanticGraph graph) {

		HashSet<IndexedWord> predicates = new HashSet<IndexedWord>();
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2]; // index count from 1,
															// also contains
															// punc.
		int rootIdx = graph.getFirstRoot().index();
		marked[rootIdx] = true;
		stack.add(rootIdx);
		List<IndexedWord> sentenceSubjects = new ArrayList<IndexedWord>();
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(graph
					.getNodeByIndex(s));
			for (SemanticGraphEdge edge : iter) {
				GrammaticalRelation gr = edge.getRelation();
				IndexedWord gov = edge.getGovernor();
				if (gr.toString().equals("nsubj")
						|| gr.toString().equals("dobj")
						|| (gr.toString().equals("prep") && gov.tag()
								.startsWith("VB"))) {
					predicates.add(edge.getGovernor());
				}

				// find sentence subject
				if (gr.toString().equals("nsubj") && gov.tag().startsWith("VB")) {
					Collection<IndexedWord> sibs = graph.getSiblings(edge
							.getDependent());
					for (IndexedWord sib : sibs) {
						GrammaticalRelation dgr = graph.reln(gov, sib);
						if (dgr.toString().equals("dobj")
								|| (dgr.toString().equals("prep"))) {

							sentenceSubjects.add(edge.getDependent());
						}
					}
				}

				int depIdx = edge.getDependent().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}

		ArrayList<InformationItem> possibleItems = new ArrayList<InformationItem>();

		if (sentenceSubjects.size() == 0)
			return possibleItems;

		for (IndexedWord p : predicates) {

			boolean subjectExist = false;
			boolean directObjectExist = false;
			boolean prepObjectExist = false;
			IndexedWord subject = null;
			IndexedWord directObject = null;
			IndexedWord prep = null;
			IndexedWord prepObject = null;

			// travel the graph

			stack = new Stack<Integer>();
			marked = new boolean[graph.size() * 2]; // index count from 1, also
													// contains punc.
			rootIdx = graph.getFirstRoot().index();
			marked[rootIdx] = true;
			stack.add(rootIdx);
			while (!stack.isEmpty()) {
				int s = stack.pop();
				Iterable<SemanticGraphEdge> iter = graph
						.outgoingEdgeIterable(p);
				for (SemanticGraphEdge edge : iter) {
					GrammaticalRelation gr = edge.getRelation();
					IndexedWord gov = edge.getGovernor();
					if (gr.toString().equals("nsubj")) {
						subjectExist = true;
						subject = edge.getDependent();
					}

					if (gr.toString().equals("dobj")) {
						directObjectExist = true;
						directObject = edge.getDependent();
					}

					if (gr.toString().equals("prep")
							&& gov.tag().startsWith("VB")) {

						Iterable<SemanticGraphEdge> children = graph
								.outgoingEdgeIterable(edge.getDependent());
						for (SemanticGraphEdge child : children) {
							GrammaticalRelation dgr = child.getRelation();
							if (dgr.toString().equals("pobj")) {
								prepObjectExist = true;
								prep = edge.getDependent();
								prepObject = child.getDependent();
							}
						}

					}

					int depIdx = edge.getDependent().index();
					if (!marked[depIdx]) {
						marked[depIdx] = true;
						stack.add(depIdx);
					}
				}
			}

			if (subjectExist == false && directObjectExist == true
					&& prepObjectExist == false) {
				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(directObject);
				possibleItems.add(new InformationItem(sentenceSubjects.get(0),
						p, obj));

			} else if (subjectExist == false && directObjectExist == false
					&& prepObjectExist == true) {

				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(sentenceSubjects.get(0),
						p, obj));

			} else if (subjectExist == true && directObjectExist == false
					&& prepObjectExist == false) {

				possibleItems.add(new InformationItem(subject, p, null));

			} else if (subjectExist == true && directObjectExist == true
					&& prepObjectExist == false) {
				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(directObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			} else if (subjectExist == true && directObjectExist == false
					&& prepObjectExist == true) {

				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			} else if (subjectExist == true && directObjectExist == true
					&& prepObjectExist == true) {
				// One Amish man craned his head out a buggy window
				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(directObject);
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			}
		}

		return possibleItems;

	}

	private ArrayList<String> generate(SemanticGraph graph) {

		SPhraseSpec newSent = nlgFactory.createClause();
		ArrayList<String> comSents = new ArrayList<String>();

		ArrayList<InformationItem> items = extractInformationItems(graph);

		if (items.size() != 0)
			for (InformationItem item : items) {

				System.out.println("Information Item is: " + item.toString());

				NPPhraseSpec subjectNp = generateNP(graph, item.getSubject());

				newSent.setSubject(subjectNp);

				VPPhraseSpec vp = generateVP(graph, item.getPredicate(),
						item.getObject());

				newSent.setVerbPhrase(vp);

				String output = realiser.realiseSentence(newSent);

				System.out.println("Generated sent is: " + output);

				comSents.add(output);
			}
		return comSents;
	}

	// search the tree recursively
	private IndexedWord searchObjforPrep(SemanticGraph graph,
			IndexedWord prepNode) {

		IndexedWord obj = null;
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2];
		int headIdx = prepNode.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		boolean stop = false;
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph
					.outgoingEdgeIterable(prepNode);

			for (SemanticGraphEdge edge : iter) {
				GrammaticalRelation dgr = edge.getRelation();
				if (dgr.toString().endsWith("obj")
						|| dgr.toString().endsWith("pcomp")) {
					obj = edge.getDependent();
					stop = true;
				}
				int depIdx = edge.getDependent().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
			if (stop == true)
				break;
		}

		return obj;
	}

	private NPPhraseSpec generateNP(SemanticGraph graph, IndexedWord head) {

		NPPhraseSpec np = nlgFactory.createNounPhrase();
		np.setHead(head.originalText());
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2];
		int headIdx = head.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(graph
					.getNodeByIndex(s));
			for (SemanticGraphEdge edge : iter) {
				if (edge.getGovernor().index() == edge.getDependent().index())
					continue; // prevent infitive recusion

				GrammaticalRelation gr = edge.getRelation();

				int depIdx = edge.getDependent().index();

				if (gr.toString().equals("prep")) {
					String prep = edge.getDependent().originalText();
					IndexedWord obj = searchObjforPrep(graph,
							edge.getDependent());
					if (obj != null) {
						PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(ppp);
						} else
							np.setPostModifier(ppp);
					}

					continue; // do not deep travel any more

				} else if (gr.toString().equals("nn")) {
					NPPhraseSpec nounModifier = generateNP(graph,
							edge.getDependent());
					if (edge.getDependent().index() < head.index()) {
						if (np.getPreModifiers().size() != 0) {
							np.addPreModifier(nounModifier);
						} else
							np.setPreModifier(nounModifier);
					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(nounModifier);
						} else
							np.setPostModifier(nounModifier);
					}

					continue;

				} else if (gr.toString().equals("conj")) {

					Iterable<SemanticGraphEdge> children = graph
							.outgoingEdgeIterable(edge.getGovernor());
					IndexedWord cc = null;
					for (SemanticGraphEdge child : children) {
						GrammaticalRelation dgr = child.getRelation();
						if (dgr.toString().equals("cc")) {
							cc = child.getDependent();
						}
					}
					NPPhraseSpec nounModifier = generateNP(graph,
							edge.getDependent());
					if (cc != null) {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(cc.originalText());
							np.addPostModifier(nounModifier);
						} else {
							np.setPostModifier(cc.originalText());
							np.addPostModifier(nounModifier);
						}
					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(nounModifier);
						} else {
							np.addPostModifier(nounModifier);
						}
					}

					continue;
				} else if (gr.toString().equals("det")
						|| gr.toString().equals("poss")) {
					IndexedWord det = edge.getDependent();
					np.setSpecifier(det.value());
				} else if (gr.toString().equals("num")) {

					IndexedWord numModifier = edge.getDependent();
					np.setSpecifier(numModifier.value());

				} else if (gr.toString().equals("amod")) {
					IndexedWord adjMod = edge.getDependent();
					if (adjMod.index() < head.index()) {
						if (np.getPreModifiers().size() != 0) {
							np.addPreModifier(adjMod.originalText());
						} else
							np.setPreModifier(adjMod.originalText());

					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(adjMod.originalText());
						} else
							np.setPostModifier(adjMod.originalText());
					}

				} else
					continue; // this is ignore all the other children

				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}// end of typed Dependency
		}

		return np;
	}

	private VPPhraseSpec generateVP(SemanticGraph graph, IndexedWord headVp,
			ArrayList<IndexedWord> object) {

		VPPhraseSpec vp = nlgFactory.createVerbPhrase();
		vp.setHead(headVp.originalText());
		// set aux of the headVerb
		Iterable<SemanticGraphEdge> children = graph
				.outgoingEdgeIterable(headVp);
		for (SemanticGraphEdge edge : children) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("aux")) {
				vp.setPreModifier(edge.getDependent().originalText());
				break;
			}
		}

		// set object
		if (object != null) {

			if (object.size() == 1) {
				// set direct object
				NPPhraseSpec dirObjNp = generateNP(graph, object.get(0));
				vp.setObject(dirObjNp);
			}

			if (object.size() == 2) {
				// set prep object from direct children
				String prep = object.get(0).originalText();
				IndexedWord obj = searchObjforPrep(graph, object.get(0));
				PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
				vp.setObject(ppp);

			}

			if (object.size() == 3) {

				// set direct and prep object
				NPPhraseSpec dirObjNp = generateNP(graph, object.get(0));
				vp.setObject(dirObjNp);

				String prep = object.get(1).originalText();
				IndexedWord obj = searchObjforPrep(graph, object.get(1));
				PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
				vp.setPostModifier(ppp);

			}

		}
		return vp;
	}

	private PPPhraseSpec generatePrepP(SemanticGraph graph, String prep,
			IndexedWord np) {
		PPPhraseSpec ppp = nlgFactory.createPrepositionPhrase();
		ppp.setPreposition(prep);
		NPPhraseSpec npp = generateNP(graph, np);
		ppp.setObject(npp);
		return ppp;
	}

	private String request(HttpMethod method) {

		String response = null;

		// Provide custom retry handler is necessary
		method.getParams().setParameter(HttpMethodParams.RETRY_HANDLER,
				new DefaultHttpMethodRetryHandler(3, false));

		try {
			// Execute the method.
			int statusCode = client.executeMethod(method);

			if (statusCode != HttpStatus.SC_OK) {
				LOG.error("Method failed: " + method.getStatusLine());
			}

			// Read the response body.
			byte[] responseBody = method.getResponseBody(); // TODO Going to
															// buffer response
															// body of large or
															// unknown size.
															// Using
															// getResponseBodyAsStream
															// instead is
															// recommended.

			// Deal with the response.
			// Use caution: ensure correct character encoding and is not binary
			// data
			response = new String(responseBody);

		} catch (HttpException e) {
			LOG.error("Fatal protocol violation: " + e.getMessage());
			System.out.println("Fatal protocol violation");
			System.exit(0);

		} catch (IOException e) {
			LOG.error("Fatal transport error: " + e.getMessage());
			LOG.error(method.getQueryString());
			System.out.println("Fatal transport error");
			System.exit(0);
		} finally {
			// Release the connection.
			method.releaseConnection();
		}
		return response;

	}

	private String NameEntityRecognition(String sentMention) {
		String spotlightResponse = null;

		GetMethod getMethod;

		try {
			getMethod = new GetMethod(API_URL + "rest/annotate/?"
					+ "confidence=" + CONFIDENCE + "&support=" + SUPPORT
					+ "&text=" + URLEncoder.encode(sentMention, "utf-8"));

			getMethod
					.addRequestHeader(new Header("Accept", "application/json"));

			spotlightResponse = request(getMethod);
			assert spotlightResponse != null;

		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return spotlightResponse;

	}

	private void extractionSerialization(String outputSummaryDir,
			String corpusName, InstanceList corpus)
			throws FileNotFoundException, IOException, ClassNotFoundException {

		for (Instance doc : corpus) {
			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();

			HashMap<CoreMap, String> ner = new HashMap<CoreMap, String>();

			Set<CoreMap> sentences = map.keySet();
			for (CoreMap sent : sentences) {
				String response = NameEntityRecognition(sent.toString());
				ner.put(sent, response);
			}
			doc.setTarget(ner);
		}

		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));
		corpus.writeObject(out);
		out.close();
	}

	private String labelVoting(String mention) {

		String[] toks = mention.split(",");
		HashMap<String, Integer> counts = new HashMap<String, Integer>();
		for (String s : toks) {
			if (s.startsWith("DBpedia")) {
				String label = s.replaceAll("DBpedia:", "");
				label = label.toLowerCase();
				if (!counts.containsKey(label)) {
					counts.put(label, 1);
				} else {
					int c = counts.get(label);
					counts.put(label, ++c);
				}

			} else if (s.startsWith("Freebase")) {
				String tmp = s.replaceAll("Freebase:/", "");
				String[] ls = tmp.split("/");
				for (String label : ls) {
					label = label.toLowerCase();
					if (!counts.containsKey(label)) {
						counts.put(label, 1);
					} else {
						int c = counts.get(label);
						counts.put(label, ++c);
					}
				}
			}
		}
		LinkedHashMap map = RankMap.sortHashMapByValues(counts, false);
		String ret = (String) map.keySet().iterator().next();
		return ret;
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
	
	private ArrayList<String> filteredArgument(String argument, CoreMap sent){
		
		SPhraseSpec newSent = nlgFactory.createClause();
		
	    SemanticGraph graph = sent.get(BasicDependenciesAnnotation.class);
	        
	    String[] words = argument.split("\\s|,");
	    ArrayList<String> npList = new ArrayList<String>();
	    for(String s : words){
	    	if(s.equals(""))
	    		continue;
	    	IndexedWord iw = graph.getNodeByWordPattern(s);
	    	if(iw == null)
	    		continue;
	    	if(iw.tag().startsWith("NN")){ 		
	    		NPPhraseSpec npSpec = generateNP(graph, iw);
	    		newSent.setSubject(npSpec);   		
	    		String np  = realiser.realiseSentence(newSent);
	    		np = np.substring(0, np.length()-1);
	    		np = np.replaceAll(",", " ");
	    		np = np.replaceAll("\\s+", " ");
	    		String head = npSpec.getHead().getRealisation();
	    		npList.add(head);
	    		npList.add(np);
	    	}
	    }	
	    	    
	    if(npList != null)
	    	return npList;
	    else{
	    	npList.add(argument);
	    	return npList;
	    }
	}

	private ArrayList<Pattern> generatePatterns(CoreMap sent,
			ArrayList<Tuple> tuples, String spotlightResponse, PrintWriter out)
			throws JSONException {
		
		//step 1: remove all tuples which relation contain said. 
		
		//how about relation extraction and headNp mutual reinforcement. 
		//first we need to noun phrase merge. 
		//noun phrase replacement by head noun only if noun phrase contains arguments.
		
		JSONObject resultJSON = null;
		JSONArray entities = null;
		resultJSON = new JSONObject(spotlightResponse);
		String originalSent = resultJSON.getString("@text");
		out.println("ORGI: "+originalSent);
		
		//for bettern match, the argument should not contain prep, 
		//and need extract head and np;
		ArrayList<Tuple> filter_tuples = new ArrayList<Tuple>();
		Set<String> argumentSet = new HashSet<String>();
		for (Tuple t : tuples) {
			if(!t.gerRel().equals("said")){
				ArrayList<String> s1 = filteredArgument(t.getArg1(), sent);
				for(String s : s1)
					argumentSet.add(s);
				
				ArrayList<String> s2 = filteredArgument(t.getArg2(), sent);	
				for(String s : s2)
					argumentSet.add(s);
	             
				filter_tuples.add(t);
			}
		}
		
		for(Tuple t : filter_tuples){
			out.println(t);
		}

		Tree tree = sent.get(TreeAnnotation.class);
		HeadFinder headFinder = new CollinsHeadFinder();
		HashMap<String, String> npheadMap = new HashMap<String, String>();
		dfs(tree, tree, headFinder, npheadMap);

        HashMap<String, Integer> npSizeMap = new HashMap<String, Integer>();
        for(String np : npheadMap.keySet()){
        	npSizeMap.put(np, np.split(" ").length);
        }
        
        LinkedHashMap<String, Integer> rankedNpSizeMap = 
        		RankMap.sortHashMapByValues(npSizeMap, false);
        
        ArrayList<String> npList = new ArrayList<String>();
        for(String np : rankedNpSizeMap.keySet()){
        	npList.add(np);
        }
        ArrayList<Integer> rmIdxList = new ArrayList<Integer>();
        for(int i=0; i<npList.size()-1; i++){
        	String np_i = npList.get(i);
        	for(int j= i+1; j<npList.size(); j++){
        		String np_j = npList.get(j);
        		if(np_i.contains(np_j) && !rmIdxList.contains(j)){
        			rmIdxList.add(j);
        		}
        	}
        }  
        ArrayList<String> mergedNpList = new ArrayList<String>();
        for(int i=0; i<npList.size(); i++){
        	if(!rmIdxList.contains(i))
        		mergedNpList.add(npList.get(i));
        }

        
		for(String np : mergedNpList){
			out.println(np+"--->"+npheadMap.get(np)+"  ");
		}
        
        String compressedSent = originalSent;
        for(String np : mergedNpList){
        	for(String arg : argumentSet){
        		if(np.toLowerCase().contains(arg.toLowerCase()) || 
        				arg.toLowerCase().contains(np.toLowerCase())){
        			compressedSent = compressedSent.replace(np, npheadMap.get(np));
        			break;
        		}
        	}	
        }
        
        out.println("COMSENT: "+compressedSent);
        	
		out.println();
		out.println();
		out.println();
		

		
/*		StringBuilder sb = new StringBuilder();
		HashMap<String, String> headStanfordNERmap = new HashMap<String, String>();
		for (CoreLabel token : sent.get(TokensAnnotation.class)) {
			String word = token.get(TextAnnotation.class);
			String pos = token.get(PartOfSpeechAnnotation.class);
			String ne = token.get(NamedEntityTagAnnotation.class);
			sb.append(word + "/" + ne + " ");
			headStanfordNERmap.put(word, ne);
		}
		System.out.println(sb.toString());
		System.out.println("_______________________________________________");*/
		
/*		HashMap<String, String> headDBpediaNERmap = new HashMap<String, String>();

		entities = resultJSON.getJSONArray("Resources");
		for (int i = 0; i < entities.length(); i++) {
			JSONObject entity = entities.getJSONObject(i);
			String types = entity.getString("@types");
			if (types.length() != 0) {
				String tok = entity.getString("@surfaceForm");
				System.out.print( tok + "-->");
				String type = labelVoting(types);
				System.out.print(type + "        " + types);
				System.out.println();
				headDBpediaNERmap.put(tok, type);
			}
		}
		System.out.println("_______________________________________________");*/
		


		return null;
	}

	private ArrayList<Pattern> patternSelection(ArrayList<Pattern> patterns) {

		return null;
	}

	private ArrayList<Pattern> generateEventSchema(String outputSummaryDir,
			String corpusName, InstanceList corpus) throws IOException,
			ClassNotFoundException, JSONException {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));

		corpus.readObject(in);
		
		PrintWriter out = FileOperation.getPrintWriter(new File(outputSummaryDir), corpusName+".ana");

		ArrayList<Pattern> allPatterns = new ArrayList<Pattern>();
		for (Instance doc : corpus) {

			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();

			HashMap<CoreMap, String> ner = (HashMap<CoreMap, String>) doc
					.getTarget();

			for (CoreMap sent : map.keySet()) {

				ArrayList<Tuple> tuples = map.get(sent);
				String spotlightResponse = ner.get(sent);
				ArrayList<Pattern> patterns = generatePatterns(sent, tuples,
						spotlightResponse, out);
				// allPatterns.addAll(patterns);
			}
		}

		out.close();
		// pattern clustering
		ArrayList<Pattern> ret = patternSelection(allPatterns);
		return ret;
	}

	private List<String> generateSummary(ArrayList<Pattern> schema) {

		return null;
	}

	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine) throws IOException,
			ClassNotFoundException, JSONException {

		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);

		InstanceList docs = new InstanceList(pipeLine);

		
/*		docs.addThruPipe(fIter);
		  
		System.out.println("Begin extraction serialization");
		extractionSerialization(outputSummaryDir, corpusName, docs);*/
		

		System.out.println("Begin generate envent schema");
		ArrayList<Pattern> schema = generateEventSchema(outputSummaryDir,
				corpusName, docs);

		/*
		 * System.out.println("Begin generate final summary"); List<String>
		 * summary = generateSummary(schema); PrintWriter out =
		 * FileOperation.getPrintWriter(new File( outputSummaryDir),
		 * corpusName); for (String sentence : summary) { out.println(sentence);
		 * } out.close();
		 */
	}

}
