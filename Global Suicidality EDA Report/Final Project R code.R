# Visualizations

# Total Suicides by Age Group Globally
ggplot(sdata_age, aes(x = year, y = total_suicides, color = age)) + 
  geom_point() + 
  scale_y_continuous(labels = comma) +
  labs(title = "Total Suicides by Age Group Globally", x = "Year", y = "Total Suicides")

# Suicide vs GDP for Japan
ggplot(subset(sdata_gdp, country %in% "Japan"), aes(x = gdp, y = total_suicides)) + geom_point(color = "darkblue") +
  labs(x = "GDP", y = "Total Suicides", title = "Suicide vs GDP for Japan") + 
  geom_smooth(method = "lm", se = FALSE, color = "darkblue")

# Suicide vs GDP for the France
ggplot(subset(sdata_gdp, country %in% "France"), aes(x = gdp, y = total_suicides)) + geom_point(color = "darkgreen") +
  labs(x = "GDP", y = "Total Suicides", title = "Suicide vs GDP for France") +
  geom_smooth(method = "lm", se = FALSE, color = "darkgreen")

# Suicide vs GDP for the Russian Federation
ggplot(subset(sdata_gdp, country %in% "Russian Federation"), aes(x = gdp, y = total_suicides)) +
  geom_point(color = "purple") +
  labs(x = "GDP", y = "Total Suicides", title = "Suicide vs GDP for the Russian Federation") +
  geom_smooth(method = "lm", se = FALSE, color = "purple")

# Suicide vs GDP for the United States
ggplot(subset(sdata_gdp, country %in% "United States"), aes(x = gdp, y = total_suicides)) +
  geom_point(color = "red") +
  labs(x = "GDP", y = "Total Suicides", title = "Suicide vs GDP for the United States") +
  geom_smooth(method = "lm", se = FALSE, color = "red")

# HDI vs Total Suicides Globally
ggplot(sdata_gdp_HDI, aes(x = HDI, y = total_suicides, color = country)) +
  geom_point() +
  theme(legend.position = "none") +
  labs(title = "HDI vs Total Suicides Globally", x = "Total Suicides", y = "HDI")

# Suicde vs GDP Globally 
ggplot(sdata_gdp, aes(x = gdp, y = total_suicides, color = country)) +
  geom_point() +
  theme(legend.position = "none") +
  labs(x = "GDP", y = "Total Suicides", title = "Suicide vs GDP Globally")

# HDI vs Total Suicides Globally (with line)
ggplot(sdata_gdp_HDI1, aes(x = HDI, y = total_suicides)) +
  geom_point(aes(color = country)) +
  theme(legend.position = "none") +
  labs(title = "HDI vs Total Suicides Globally", x = "Total Suicides", y = "HDI") +
  geom_smooth(method = lm, se = FALSE)

# Suicide vs GDP Globally (with line)
ggplot(sdata_gdp1, aes(x = gdp, y = total_suicides)) +
  geom_point(aes(color = country)) + theme(legend.position = "none") +
  labs(x = "GDP", y = "Total Suicides", title = "Suicide vs GDP Globally") +
  geom_smooth(method = "lm", se = FALSE)

# Total Suicides by Gender Globally
ggplot(sdata_bysex, aes(x = sex, y = total_suicides, fill = sex)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(labels = comma) +
  labs(title = "Total Suicides by Gender Globally", x = "Gender", y = "Total Suicides") +
  theme(legend.position = "none")

# Total Suicides by Generation Globally
ggplot(sdata_gen, aes(x = year, y = total_suicides, color = generation)) +
  geom_point() +
  geom_line() +
  labs(title = "Total Suicides by Generation Globally", x = "Year", y = "Total Suicides")

# Suicide Rate Globally
ggplot(sdata3_trend, aes(x = year, y = suicide_rate)) +
  geom_point(color = "blue") +
  geom_line(color = "blue") +
  labs(title = "Suicide Rate Globally", x = "Year", y = "Suicide Rate")

# Sucide Rate for the US
ggplot(sdata_us_sum, aes(x = year, y = suicide_rate)) +
  geom_point(color = "red") +
  geom_line(color = "red") +
  labs(title = "Suicide Rate for the US", x = "Year", y = "Suicide Rate")

# Total Suicides for the US
ggplot(sdata_us_sum, aes(x = year, y = suicide_rate)) +
  geom_point(color = "red") +
  geom_line(color = "red") +
  labs(title = "Suicide Rate for US", x = "Year", y = "Suicide Rate")

# Datasets

# sdata_bysex (For Total Suicides by Gender)
sdata_bysex <- sdata3 %>%
  group_by(year, sex) %>% 
  summarise(total_suicides = sum(suicides_no)) %>%
  distinct(year, total_suicides, sex)

# sdata_gen (For Total Suicides by Generation)
sdata_gen <- sdata3 %>%
group_by(year, generation) %>%
  summarise(total_suicides = sum(suicides_no))

# sdata_us_sum (For Suicides in US)
 sdata_us <- sdata3 %>%
       filter(country == "United States")
 sdata_us_sum <- sdata_us %>%
       group_by(year) %>%
       summarise(total_suicides = sum(suicides_no), total_population = sum(population)) %>%
       mutate(suicide_rate = total_suicides/total_population)
 
# sdata3_trend
sdata3_trend <- sdata3 %>%
  group_by(year) %>%
     summarise(total_suicides = sum(suicides_no), total_population = sum(population)) %>%
     mutate(suicide_rate = total_suicides/total_population)

# sdata_us_2000
sdata_us_2000 <- sdata_us %>%
  filter(year >= 2000)

# sdata_age
sdata_age <- sdata3 %>%
  group_by(age, year) %>%
  summarise(total_suicides = sum(suicides_no))

# sdata_gdp
sdata_gdp <- sdata3 %>%
  group_by(country, year) %>%
  summarise(total_suicides = sum(suicides_no), gdp = gdp_for_year) %>%
  distinct(country, year, .keep_all = TRUE)

# sdata_HDI
sdata_gdp_HDI <- sdata_HDI %>%
  group_by(country, year) %>%
  summarise(total_suicides = sum(suicides_no), gdp = gdp_for_year, HDI = HDI.for.year) %>%
  distinct(country, year, .keep_all = TRUE)

# sdata_gdp_HDI1
sdata_gdp_HDI <- sdata_gdp_HDI %>%
  filter(total_suicides < 20000)

# sdata_gdp1
sdata_gdp1 <- sdata_gdp %>%
  filter(country != "United States" & country != "Russian Federation" & country != "Ukraine")
