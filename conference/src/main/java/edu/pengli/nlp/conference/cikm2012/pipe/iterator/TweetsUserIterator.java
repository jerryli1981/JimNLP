package edu.pengli.nlp.conference.cikm2012.pipe.iterator;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.pengli.nlp.conference.cikm2012.types.Tweet;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.Token;
import edu.pengli.nlp.platform.util.FileOperation;

public class TweetsUserIterator implements Iterator<Instance> {

	Iterator<Instance> tUsersIter;

	public TweetsUserIterator(String outputDir, String fileName) {
		BufferedReader in = FileOperation.getBufferedReader(
				new File(outputDir), fileName);
		ArrayList<Instance> users = new ArrayList<Instance>();
		String input = null;
		ArrayList<Tweet> sentList = new ArrayList<Tweet>();
		Pattern p = Pattern.compile("<UserName>.*?</UserName>");
		Pattern pp = Pattern.compile("<RelTweet>.*?</RelTweet>");
		HashMap<String, ArrayList<Tweet>> userNameTwMap = new HashMap<String, ArrayList<Tweet>>();
		try {
			while ((input = in.readLine()) != null) {
				Matcher m = p.matcher(input);
				Matcher mm = pp.matcher(input);
				String username = null;
				String tweetMention = null;
				Tweet t = null;
				if (m.find()) {
					username = m.group().replaceAll("<.*?>", "");
				}
				if (mm.find()) {
					tweetMention = mm.group().replaceAll("<.*?>", "");
					String[] toks = tweetMention.split(" ");
					t = new Tweet();
					for (int i = 0; i < toks.length; i++) {
						Token tok = new Token(toks[i]);
						t.add(i, tok);
					}
				}
				if(username == null || tweetMention == null){
					System.out.println("username or tweetMention is null");
			    	System.out.println(input);
					System.exit(0);
				}

				if (!userNameTwMap.containsKey(username)) {
					sentList = new ArrayList<Tweet>();
					sentList.add(t);
					userNameTwMap.put(username, sentList);
				} else {
					sentList = userNameTwMap.get(username);
					sentList.add(t);
					userNameTwMap.put(username, sentList);
				}

			}//end while
			
			
			Set<String> userNames = userNameTwMap.keySet();
			Iterator iter = userNames.iterator();
			while(iter.hasNext()){
				Instance user = new Instance(null,null,null);
				String username = (String) iter.next();
				sentList = userNameTwMap.get(username);
				StringBuffer tweetMention = new StringBuffer();
				for(Tweet t : sentList){
					tweetMention.append(t.toString()+"\n");
				}
				user.setName(username);
				user.setData(tweetMention.toString().trim());
				users.add(user);
			}

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		tUsersIter = users.iterator();
	}

	@Override
	public boolean hasNext() {
		// TODO Auto-generated method stub
		return tUsersIter.hasNext();
	}

	@Override
	public Instance next() {
		// TODO Auto-generated method stub
		Instance user = tUsersIter.next();
		return new Instance(user, null, user.getName()+"_T", null);
	}

	@Override
	public void remove() {
		throw new IllegalStateException("Not supported");
	}

}