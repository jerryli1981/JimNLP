package edu.pengli.nlp.conference.cikm2012.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.Version;

import cc.twittertools.index.IndexStatuses;

import edu.pengli.nlp.conference.cikm2012.types.Topic;
import edu.pengli.nlp.platform.util.FileOperation;

public class GenerateTestDocuments {

	public static String retrievalMention(File indexLocation, String id)
			throws Exception {

		File[] fl = indexLocation.listFiles();
		String mention = null;
		for (File f : fl) {
			Directory dir = new MMapDirectory(f);
			IndexReader reader = IndexReader.open(dir);
			IndexSearcher searcher = new IndexSearcher(reader);
			QueryParser qparser = new QueryParser(Version.LUCENE_35,
					IndexStatuses.StatusField.ID.name, IndexStatuses.ANALYZER);
			Query query = qparser.parse(id);

			TopDocs rs = searcher.search(query, 10);
			if (rs.totalHits == 0)
				continue;
			for (ScoreDoc scoreDoc : rs.scoreDocs) {
				Document hit = searcher.doc(scoreDoc.doc);
	
				mention = hit.getField(IndexStatuses.StatusField.TEXT.name)
						.stringValue();
				String userid = hit.getField(
						IndexStatuses.StatusField.SCREEN_NAME.name)
						.stringValue();
				mention = userid + "<::>" + mention;
			}
			if (rs.totalHits == 1) {
				break;
			}
		}

		return mention;
	}

	public static void main(String[] args) throws Exception {
		// read topic list
		File dataDir = new File(
				"../data/EMNLP2012");
		String topicFileName = "topics.MB1-50.txt";
		String content = FileOperation.readContentFromFile(dataDir,
				topicFileName);
		content = content.replaceAll("\n", "");
		ArrayList<Topic> topicList = new ArrayList<Topic>();
		Pattern p = Pattern.compile("<top>.*?</top>");
		Matcher m = p.matcher(content);
		while (m.find()) {
			String topStr = m.group();
			Topic topic = new Topic(topStr);
			topicList.add(topic);
		}

		HashMap<Integer, ArrayList<String>> topIdxRelevantTweetsIdMap = new HashMap<Integer, ArrayList<String>>();
		String qrelFileName = "microblog11-qrels.txt";
		BufferedReader in = FileOperation.getBufferedReader(dataDir,
				qrelFileName);
		String input = null;
		ArrayList<String> relTweetsIds = null;
		int lastIdx = Integer.MAX_VALUE;
		while ((input = in.readLine()) != null) {
			String[] items = input.split(" ");
			int idx = Integer.parseInt(items[0]);
			if (topIdxRelevantTweetsIdMap.isEmpty()) {
				topIdxRelevantTweetsIdMap.put(idx, null);
				relTweetsIds = new ArrayList<String>();
			}
			if (!topIdxRelevantTweetsIdMap.containsKey(idx)) {
				topIdxRelevantTweetsIdMap.put(lastIdx, relTweetsIds);
				relTweetsIds = new ArrayList<String>();
				topIdxRelevantTweetsIdMap.put(idx, null);
			}

			int score = Integer.parseInt(items[3]);
			if (score > 0)
				relTweetsIds.add(items[2]);
			lastIdx = idx;

		}
		if ((input = in.readLine()) == null) {
			topIdxRelevantTweetsIdMap.put(lastIdx, relTweetsIds);
		}

		File indexLocation = new File(
				"../data/EMNLP2012/INDEX");

		for (Topic t : topicList) {
			int id = Integer.parseInt(t.getID().replace("Number: MB0", ""));
			System.out.println(id); 
			File f = new File(
					"../data/EMNLP2012/Topics/Twitter");
			PrintWriter out = FileOperation.getPrintWriter(f,
					String.valueOf(id));
			ArrayList<String> relTsIds = topIdxRelevantTweetsIdMap.get(id);
			// out.println("<Title>"+t.getTitle()+"</Title>");
			/*
			 * System.out.println(t.getID() + " : " + t.getTitle() + " " +
			 * relTsIds.size());
			 */
			for (String idx : relTsIds) {
				String rel = retrievalMention(indexLocation, idx);
				if (rel == null)
					continue;
				else{
				String userid = rel.split("<::>")[0];
				// if(userid == null) continue;
				String mention = rel.split("<::>")[1];
		
				out.println("<UserName>" + userid + "</UserName><RelTweet>"
						+ mention + "</RelTweet>");
				}

			}
			out.close();
		}

		// 29 tweets are missing
		System.out.println("done");

	}

}
