package edu.pengli.nlp.conference.cikm2015.pipe;


import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.pengli.nlp.platform.pipe.CharSequenceRemoveHTML;
import edu.pengli.nlp.platform.types.Instance;

public class CharSequenceExtractContent extends CharSequenceRemoveHTML {
	
	public CharSequenceExtractContent(String regex){
		super(regex);
	}
	
	private String fixPuncAndFilterLength(String content){
		String[] sents = content.split("\\n");
		StringBuilder sb  = new StringBuilder();
		for(String sent : sents){
			String[] toks = sent.split(" ");
			if(toks.length <= 5)
				continue;
			StringBuilder lastChar = new StringBuilder();
			StringBuilder tmpSent = new StringBuilder();
			lastChar.append(sent.charAt(sent.length()-1));
			if(!lastChar.toString().matches("\\p{Punct}")){
				tmpSent.append(sent+".");
			}else{
				if(!lastChar.toString().equals(".")){
					tmpSent.append(sent.substring(0, sent.length()-1)+".");
				}else{
					tmpSent.append(sent);
				}
			}
			sb.append(tmpSent.toString().trim()+"\n");
		}
		return sb.toString().trim();
	}
	public Instance pipe(Instance carrier) {
		//<s docid="APW_ENG_20070615.0356" num="2" stype="1">FORT LAUDERDALE, Florida 2007-06-15 05:06:17 UTC</s>PARA
		String content = carrier.getData().toString();
       
        Pattern fil = Pattern.compile(">.*\\s(UTC)");
        Matcher mfil = fil.matcher(content);
        if(mfil.find()){
        	String find = mfil.group();
        	content = content.replace(find, ">");
        }
        
        Matcher m = p.matcher(content);
        if(m.find()){
        	content = m.group();
        	content = content.replaceAll("</s>PARA", "</s>");
        	content = content.replaceAll("<.*?>", "");
        	content = content.replaceAll("\"", "");
        	content = content.replaceAll("\\s?\\(.*?\\)\\s?", " ");
        	content = content.replaceAll("'ll", " will");
        	content = content.replaceAll("``", "");
        	content = content.replaceAll("&amp;amp;", "&");
        	content = content.replaceAll("&amp;", "&");
        	content = content.replaceAll("&amp", "&");
        	content = content.trim();
        	content = fixPuncAndFilterLength(content);
        	
        }
		carrier.setData((CharSequence) content);
		return carrier;
	}

}
