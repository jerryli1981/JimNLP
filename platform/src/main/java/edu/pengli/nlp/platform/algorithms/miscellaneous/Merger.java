package edu.pengli.nlp.platform.algorithms.miscellaneous;


import java.util.ArrayList;

public class Merger {

	private static boolean contains(String sent, String thisSent) {
		String[] sentList = sent.split(" ");
		String[] thisList = thisSent.split(" ");
		if (thisList.length < sentList.length) {
			return false;
		} else {
			int si = 0;
			int tj = 0;
			for (int i = 0; i < sentList.length; i++) {
				String ei = sentList[i];
				for (int j = 0; j < thisList.length; j++) {
					if (tj < thisList.length) {
						String ej = thisList[tj];
						if (ei.equals(ej)) {
							tj++;
							si++;
							break;
						} else {
							tj++;
						}
					}

				}
			}
			if (si == sentList.length) {
				return true;
			}

		}
		return false;

	}

	public static ArrayList<String> process(ArrayList<String> originalList) {
		
		ArrayList<String> mergedList = new ArrayList<String>();
		for (int i = 0; i < originalList.size(); i++) {
			String si = originalList.get(i);
			boolean flag = false;
			for (int j = 0; j < originalList.size(); j++) {
				if (i == j)
					continue;
				String sj = originalList.get(j);
				if (contains(si, sj)) {
					flag = true;
					break;
				}
			}
			if (flag == false) {
				mergedList.add(si);
			}
		}
		
		return mergedList;
	}
	
	public static void main(String[] args){
		String s1 = "More than 6,000 houses were without power on Monday in Los Angeles , where county officials estimate the property damage in excess of 19 million";
		String s2 = "More than 6,000 houses were without power on Monday in Los Angeles , where county officials estimate the property damage is in excess of 19 million";
		String s3 = "More than 6,000 houses were without power on Monday in Los Angeles , county officials estimate the property damage is in excess of 19 million";
		String s4 = "More than 6,000 houses without power on Monday in Los Angeles , where county officials estimate the property damage in excess of 19";
		ArrayList<String> test = new ArrayList<String>();
		test.add(s1);
		test.add(s2);
		test.add(s3);
		test.add(s4);
		ArrayList<String> tmp = process(test);
		System.out.println(tmp.size());
		System.out.println(tmp);
	}

}
