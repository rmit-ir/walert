require(tidyverse)
require(stringr)
#Read data


baseline  <- read.csv("/Users/mkazemi/Library/CloudStorage/OneDrive-RMITUniversity/My Mac Folders/2024/RACE EIP/Damiano/walert/quantitative_eval/target/summaries/walert_eval.csv")
rag.bm25  <- read.csv("/Users/mkazemi/Library/CloudStorage/OneDrive-RMITUniversity/My Mac Folders/2024/RACE EIP/Damiano/falcon_results_metrics.csv")
rag.dense <- read.csv("/Users/mkazemi/Library/CloudStorage/OneDrive-RMITUniversity/My Mac Folders/2024/RACE EIP/Damiano/walert/quantitative_eval/target/summaries/falcon_dense_eval.csv")


metrics <- c("bert_score_f1","bleu_score", "rouge_l_f1","rouge_1_f1","rouge_2_f1")


sets <- c("top3")

for (metric in metrics) {
  

  cat(sprintf("walert-intent %s: %0.04f\n", metric, baseline %>% summarise(mean(!!sym(metric)))))
  for (set in sets) {
    cat(sprintf("rag-bm25(%s) %s: %0.04f\n",set, metric, rag.bm25 %>% summarise(mean(!!sym(paste(metric,set,sep = "_"))))))
    cat(sprintf("rag-dense(%s) %s: %0.04f\n",set, metric, rag.dense %>% summarise(mean(!!sym(paste(metric,set, sep ="_"))))))
  }
}

# System: intent-based rag-bm25-top1 rag-bm25-top3 rag-bm25-top5 rag-dense-top1 rag-dense-top3 rag-dense-top5

# Score_BERTScore Score_BLEU Score_ROUGE_1 Score_ROUGE_2 Score_ROUGE_l


data <- baseline %>% select(question_id,approach,bert_score_f1,bleu_score,rouge_l_f1,rouge_1_f1,rouge_2_f1)
# 
# temp <- rag.bm25 %>% select(question_id,bert_score_f1_top1,bleu_score_top1,rouge_l_f1_top1,rouge_1_f1_top1,rouge_2_f1_top1)
# temp$approach <- "rag.bm25.top1"
# names(temp) <- c("question_id", "bert_score_f1","bleu_score","rouge_l_f1", "rouge_1_f1"    ,"rouge_2_f1","approach")
# data <- bind_rows(data,temp)


temp <- rag.bm25 %>% select(question_id,bert_score_f1_top3,bleu_score_top3,rouge_l_f1_top3,rouge_1_f1_top3,rouge_2_f1_top3)
temp$approach <- "rag.bm25.top3"
names(temp) <- c("question_id", "bert_score_f1","bleu_score","rouge_l_f1", "rouge_1_f1"    ,"rouge_2_f1","approach")
data <- bind_rows(data,temp)
# 
# temp <- rag.bm25 %>% select(question_id,
#                             bert_score_f1_top5,
#                             bleu_score_top5,
#                             rouge_l_f1_top5,
#                             rouge_1_f1_top5,
#                             rouge_2_f1_top5)
# temp$approach <- "rag.bm25.top5"
# names(temp) <- c("question_id", "bert_score_f1","bleu_score","rouge_l_f1", "rouge_1_f1"    ,"rouge_2_f1","approach")
# data <- bind_rows(data,temp)



# temp <- rag.dense %>% select(question_id,bert_score_f1_top1,bleu_score_top1,rouge_l_f1_top1,rouge_1_f1_top1,rouge_2_f1_top1)
# temp$approach <- "rag.dense.top1"
# names(temp) <- c("question_id", "bert_score_f1","bleu_score","rouge_l_f1", "rouge_1_f1"    ,"rouge_2_f1","approach")
# data <- bind_rows(data,temp)


temp <- rag.dense %>% select(question_id,bert_score_f1_top3,bleu_score_top3,rouge_l_f1_top3,rouge_1_f1_top3,rouge_2_f1_top3)
temp$approach <- "rag.dense.top3"
names(temp) <- c("question_id", "bert_score_f1","bleu_score","rouge_l_f1", "rouge_1_f1"    ,"rouge_2_f1","approach")
data <- bind_rows(data,temp)
# 
# temp <- rag.dense %>% select(question_id,
#                             bert_score_f1_top5,
#                             bleu_score_top5,
#                             rouge_l_f1_top5,
#                             rouge_1_f1_top5,
#                             rouge_2_f1_top5)
# temp$approach <- "rag.dense.top5"
# names(temp) <- c("question_id", "bert_score_f1","bleu_score","rouge_l_f1", "rouge_1_f1"    ,"rouge_2_f1","approach")
# data <- bind_rows(data,temp)




#W01-W20,W39	known
#W21-W32	inferred
#[W33-W38,W40-W43]	out of knowledge base

#approaches <- unique(data$approach)
#data$approach <- factor(data$approach, levels = approaches, ordered = TRUE)

# Assuming your data frame is called df
topics.known <- data %>%
  filter(startsWith(question_id, prefix = "W01") |
           startsWith(question_id, prefix = "W02") |
           startsWith(question_id, prefix = "W03") |
           startsWith(question_id, prefix = "W04") |
           startsWith(question_id, prefix = "W05") |
           startsWith(question_id, prefix = "W06") |
           startsWith(question_id, prefix = "W07") |
           startsWith(question_id, prefix = "W08") |
           startsWith(question_id, prefix = "W09") |
           startsWith(question_id, prefix = "W10") |
           startsWith(question_id, prefix = "W11") |
           startsWith(question_id, prefix = "W12") |
           startsWith(question_id, prefix = "W13") |
           startsWith(question_id, prefix = "W14") |
           startsWith(question_id, prefix = "W15") |
           startsWith(question_id, prefix = "W16") |
           startsWith(question_id, prefix = "W17") |
           startsWith(question_id, prefix = "W18") |
           startsWith(question_id, prefix = "W19") |
           startsWith(question_id, prefix = "W39") |
           startsWith(question_id, prefix = "W20")
  )


topics.inferred <- data %>%
  filter(startsWith(question_id, prefix = "W21") |
           startsWith(question_id, prefix = "W22") |
           startsWith(question_id, prefix = "W23") |
           startsWith(question_id, prefix = "W24") |
           startsWith(question_id, prefix = "W25") |
           startsWith(question_id, prefix = "W26") |
           startsWith(question_id, prefix = "W27") |
           startsWith(question_id, prefix = "W28") |
           startsWith(question_id, prefix = "W29") |
           startsWith(question_id, prefix = "W30") |
           startsWith(question_id, prefix = "W31") |
           startsWith(question_id, prefix = "W32")
  )


topics.out.of.KB <- data %>% filter(  startsWith(question_id, prefix = "W33") |
                                      startsWith(question_id, prefix = "W34") |
                                      startsWith(question_id, prefix = "W35") |
                                      startsWith(question_id, prefix = "W36") |
                                      startsWith(question_id, prefix = "W37") |
                                      startsWith(question_id, prefix = "W38") |
                                      startsWith(question_id, prefix = "W40") |
                                      startsWith(question_id, prefix = "W41") |
                                      startsWith(question_id, prefix = "W42") |
                                      startsWith(question_id, prefix = "W43")
)


#metrics <- c("bert_score_f1","bleu_score","rouge_l_f1","rouge_1_f1","rouge_2_f2")
metrics <- c("bert_score_f1","rouge_1_f1")

for (metric in metrics) {
  formula <- as.formula(paste(metric, "~ factor(approach) + factor(question_id)"))
  anova_result <- aov(formula, data = topics.known)
  tukey_result <- TukeyHSD(anova_result, "factor(approach)")
  print(metric)
  result <- data.frame(tukey_result$`factor(approach)`)
  #result$comparison <- rownames(result)
  
  stats <- result %>%  filter(p.adj < 0.05)
  stats$p_value_min <- stats$p.adj < 0.01 
  scores <- topics.known %>% group_by(approach) %>% summarise(mean(!!sym(metric)))
  print(deparse(substitute(topics.known)))
  print(metric)
  print(scores)
  print(stats[,4:5])
}

for (metric in metrics) {
  formula <- as.formula(paste(metric, "~ factor(approach) + factor(question_id)"))
  anova_result <- aov(formula, data = topics.inferred)
  tukey_result <- TukeyHSD(anova_result, "factor(approach)")
  print(metric)
  result <- data.frame(tukey_result$`factor(approach)`)
  #result$comparison <- rownames(result)
  
  stats <- result %>%  filter(p.adj < 0.05)
  stats$p_value_min <- stats$p.adj < 0.01 
  scores <- topics.inferred %>% group_by(approach) %>% summarise(mean(!!sym(metric)))
  print(deparse(substitute(topics.inferred)))
  print(metric)
  print(scores,sep="\t")
  print(stats[,4:5])
}

for (metric in metrics) {
  formula <- as.formula(paste(metric, "~ factor(approach) + factor(question_id)"))
  anova_result <- aov(formula, data = topics.out.of.KB)
  tukey_result <- TukeyHSD(anova_result, "factor(approach)")
  result <- data.frame(tukey_result$`factor(approach)`)
  #result$comparison <- rownames(result)
  
  stats <- result %>%  filter(p.adj < 0.05)
  stats$p_value_min <- stats$p.adj < 0.01 
  scores <- topics.out.of.KB %>% group_by(approach) %>% summarise(mean(!!sym(metric)))
  print(deparse(substitute(topics.out.of.KB)))
  print(metric)
  print(scores)
  print(stats[,4:5])
  
}

